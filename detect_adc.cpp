#include <Arduino.h>

#define RED_LED 22
#define GREEN_LED 23

// SAADC Constants
#define SAMPLES_PER_SECOND  16000
#define PPI_CHANNEL         (7)
#define ADC_BUFFER_SIZE     100

#define CLASSIFICATION_CUTOFF 0.70  // Threshold to consider positive detection of asthma event.
#define HOLD_TIME 4     // How long to let timer run while counting number of asthma attacks. [Seconds]
                        // Max is given by 2^16 / 2^12 (2^Timer Bitwidth / 2^Timer Prescaler)
//might need to change
#define NOTIFY_AMOUNT 3 // How many asthma occurences required to trigger notification.

#define SYSTEM_ON_TIME 10 // How long [seconds] the arduino turns on for once it wakes up
#define D9_GPIO_PIN 27

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 3

/* Includes ---------------------------------------------------------------- */
#include <dog-asthma-detection_inferencing.h>
#include <ArduinoBLE.h>
#include <stdio.h>
#include "mbed.h"
#include "rtos.h"

// ADC and PPI
volatile nrf_saadc_value_t adcBuffer[2][ADC_BUFFER_SIZE];
volatile bool buff_full_flag = false;
volatile uint8_t dbl_buff_sel = 0;
volatile uint32_t runTime = 0;
void initPPI();
void initADC();

/** Define Threads */
rtos::Thread threadRedLED(osPriorityAboveNormal);
rtos::Thread threadNotify(osPriorityAboveNormal1);

/** Timer Functions & counter */
void initTimer4();
volatile bool timer_running = false;
volatile int asthma_count = 0;

/** System Off Timer */
void initTimer3();
void initGPIOTE();
void resetTimerISR();

/** Define UUID in Bluetooth service */
BLEService respiService("19B10011-E8F2-537E-4F6C-D104768A1214");

/** Define Tasks */
void RedLED();
void Notify();

volatile int notify_flag = 0;

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

void ei_printf(const char *format, ...);
static void pdm_data_ready_inference_callback(void);
static bool microphone_inference_start(uint32_t n_samples);
static bool microphone_inference_record(void);
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
static void microphone_inference_end(void);

/**
 * @brief      Arduino setup function
 */
void setup()
{
    pinMode(LED_PWR, OUTPUT);
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    digitalWrite(LED_PWR, LOW); // Turn off power led to reduce current draw
    digitalWrite(RED_LED, HIGH); //High is off for onboard LEDs
    digitalWrite(GREEN_LED, HIGH);

    Serial.begin(115200);

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }

    // Init Timer4, PPI, and SAADC
    initTimer4();
    initPPI();
    initADC();

    // Init System Off Timer and DETECT
    initTimer3();
    initGPIOTE();

    //threadClassify.start(Classify);
    threadRedLED.start(RedLED);
    threadNotify.start(Notify);

    // Start system off timer
    NRF_TIMER3->TASKS_START = 1;
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    /****************************************
     * If ADC Buffer is full, switch buffer *
     ****************************************/
    if ( buff_full_flag==true )
    {
      buff_full_flag = false;
      NRF_SAADC->RESULT.PTR = ( uint32_t )&adcBuffer[dbl_buff_sel]; //Switch buffer
    }

    /*************
     * Inference *
     *************/
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        // print the predictions
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.5f\n", result.classification[ix].label,
                      result.classification[ix].value);
        }
        if(result.classification[0].value >= CLASSIFICATION_CUTOFF) {
                // Reset timer to 0 so that timer starts on newest asthma event.
                // This ensures that the detections are strung together with a separation
                // of HOLD_TIME.
                NRF_TIMER4->TASKS_CLEAR = 1;

                // Flash Red LED
                threadRedLED.flags_set(0x1);

                //Start timer
                if(asthma_count == 0 && timer_running == false) {
                    timer_running = true;
                    NRF_TIMER4->TASKS_START = 1;
                }

                asthma_count++;
                // Flash Green LED if conditions are met
                if(asthma_count == NOTIFY_AMOUNT && timer_running == true) {
                    asthma_count = 0;
                    threadNotify.flags_set(0x1);
                }
            }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif

        print_results = 0;
    }

    /********************
     * BLE Notification *
     ********************/
    if(notify_flag) {
        notify_flag = 0;

        if (!BLE.begin()) {
        Serial.println("- Starting BLE module failed!");
        while (1);
        }

        BLE.setLocalName("RespiPet Device");
        BLE.setDeviceName("RespiPet Device"); 

        BLE.setAdvertisedService(respiService);
        BLE.addService(respiService);

        if(BLE.advertise()) {
            Serial.println("Advertise Success");
        } else {
            Serial.println("Advertise Failure");
        }

        BLEDevice central = BLE.central();
        if (central) {
            while (central.connected()) {
                rtos::ThisThread::sleep_for(5000); // wait
                central.disconnect();
            }
        }
        
        rtos::ThisThread::sleep_for(2500);
        BLE.stopAdvertise();
        BLE.end();
    }
}

/******************************
 * Interrupt Service Routines *
 ******************************/

// Reset system off timer if sound is detected when in system on mode
void resetTimerISR()
{
  NRF_TIMER3->TASKS_CLEAR = 1; // clears timer 
}

// ISR TIMER4
extern "C" void TIMER4_IRQHandler_v( void )
{
    if (NRF_TIMER4->EVENTS_COMPARE[0] == 1)
    {   
        timer_running = false;
        asthma_count = 0;
        NRF_TIMER4->TASKS_STOP = 1;
        NRF_TIMER4->TASKS_CLEAR = 1;
        NRF_TIMER4->EVENTS_COMPARE[0] = 0;
    }
}

// ISR TIMER3
extern "C" void TIMER3_IRQHandler_v( void )
{
    if (NRF_TIMER3->EVENTS_COMPARE[0] == 1)
    {   
        NRF_TIMER3->TASKS_STOP = 1;
        NRF_TIMER3->TASKS_CLEAR = 1;
        NRF_TIMER3->EVENTS_COMPARE[0] = 0;

        NRF_POWER->SYSTEMOFF = 1;
    }
}

// ISR SAADC
extern "C" void SAADC_IRQHandler_v( void )
{
  NRF_SAADC->EVENTS_END = 0;
  buff_full_flag = true;
  dbl_buff_sel ^= 1;

  // Copy samples from DMA buffer to inference buffer
  if (record_ready) {
    for (uint32_t i = 0; i < ADC_BUFFER_SIZE; i++) {
  
      // Convert 12-bit unsigned ADC value to 16-bit PCM (signed) audio value
      inference.buffers[inference.buf_select][inference.buf_count++] =
          ((int16_t)adcBuffer[dbl_buff_sel][i] - 2048) * 16;
  
      // Swap double buffer if necessary
      if (inference.buf_count >= inference.n_samples) {
        inference.buf_select ^= 1;
        inference.buf_count = 0;
        inference.buf_ready = 1;
      }
    }
  }
}

/*********
 * Tasks *
 *********/

// Light Red LED
void RedLED () {
  while(1) {
    rtos::ThisThread::flags_wait_any(0x1);
    digitalWrite(RED_LED, LOW);
    rtos::ThisThread::sleep_for(50); // wait
    digitalWrite(RED_LED, HIGH);
    rtos::ThisThread::sleep_for(50); // wait
  }
}

// Light Green LED and send nofification via BLE
void Notify () {
  while(1) {
    
    rtos::ThisThread::flags_wait_any(0x1);
    notify_flag = 1;
    Serial.println("Notify started");
    digitalWrite(GREEN_LED, LOW);
    rtos::ThisThread::sleep_for(50); // wait
    digitalWrite(GREEN_LED, HIGH);
    rtos::ThisThread::sleep_for(50);

  }


}

/********************
 * Peripheral Inits *
 ********************/

void initTimer4()
{
  NRF_TIMER4->MODE = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->BITMODE = TIMER_BITMODE_BITMODE_24Bit;
  NRF_TIMER4->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Enabled << TIMER_SHORTS_COMPARE0_CLEAR_Pos;
  NRF_TIMER4->PRESCALER = 10;
  NRF_TIMER4->CC[0] = 15625 * 2 * HOLD_TIME; // Needs prescaler set to 10 (15625 = 16MHz / 2^10) Cant be larger than 2^16

  NRF_TIMER4->INTENSET = TIMER_INTENSET_COMPARE0_Enabled << TIMER_INTENSET_COMPARE0_Pos;
  NVIC_EnableIRQ( TIMER4_IRQn );
}


void initTimer3()
{
  NRF_TIMER3->MODE = TIMER_MODE_MODE_Timer;
  NRF_TIMER3->TASKS_CLEAR = 1;
  NRF_TIMER3->BITMODE = TIMER_BITMODE_BITMODE_24Bit;
  NRF_TIMER3->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Enabled << TIMER_SHORTS_COMPARE0_CLEAR_Pos;
  NRF_TIMER3->PRESCALER = 10;
  NRF_TIMER3->CC[0] = 15625 * 2 * SYSTEM_ON_TIME; // Needs prescaler set to 10 (15625 = 16MHz / 2^10) Cant be larger than 2^16

  NRF_TIMER3->INTENSET = TIMER_INTENSET_COMPARE0_Enabled << TIMER_INTENSET_COMPARE0_Pos;
  NVIC_EnableIRQ( TIMER3_IRQn );
}


void initGPIOTE()
{
  attachInterrupt(digitalPinToInterrupt(D9), resetTimerISR, RISING);
  
  // Note: The function below only works on port P0 pins. For P1 pins you must set the pinMode
  // and configure sense with NRF_P1->PIN_CFG 
  nrf_gpio_cfg_sense_input(D9_GPIO_PIN, NRF_GPIO_PIN_PULLUP, NRF_GPIO_PIN_SENSE_HIGH);
}


void initADC()
{
  nrf_saadc_disable();

  NRF_SAADC->RESOLUTION = NRF_SAADC_RESOLUTION_12BIT;

  NRF_SAADC->CH[2].CONFIG = ( SAADC_CH_CONFIG_GAIN_Gain1_4    << SAADC_CH_CONFIG_GAIN_Pos ) |
                            ( SAADC_CH_CONFIG_MODE_SE         << SAADC_CH_CONFIG_MODE_Pos ) |
                            ( SAADC_CH_CONFIG_REFSEL_VDD1_4   << SAADC_CH_CONFIG_REFSEL_Pos ) |
                            ( SAADC_CH_CONFIG_RESN_Bypass     << SAADC_CH_CONFIG_RESN_Pos ) |
                            ( SAADC_CH_CONFIG_RESP_Bypass     << SAADC_CH_CONFIG_RESP_Pos ) |
                            ( SAADC_CH_CONFIG_TACQ_3us        << SAADC_CH_CONFIG_TACQ_Pos );

  NRF_SAADC->CH[2].PSELP = SAADC_CH_PSELP_PSELP_AnalogInput2 << SAADC_CH_PSELP_PSELP_Pos;
  NRF_SAADC->CH[2].PSELN = SAADC_CH_PSELN_PSELN_NC << SAADC_CH_PSELN_PSELN_Pos;

  NRF_SAADC->RESULT.MAXCNT = ADC_BUFFER_SIZE;
  NRF_SAADC->RESULT.PTR = ( uint32_t )&adcBuffer[0];

  NRF_SAADC->SAMPLERATE = (SAADC_SAMPLERATE_MODE_Timers << SAADC_SAMPLERATE_MODE_Pos)
                          | ((uint32_t)(16000000 / SAMPLES_PER_SECOND) << SAADC_SAMPLERATE_CC_Pos);

  NRF_SAADC->EVENTS_END = 0;
  nrf_saadc_int_enable( NRF_SAADC_INT_END );
  NVIC_SetPriority( SAADC_IRQn, 1UL );
  NVIC_EnableIRQ( SAADC_IRQn );

  nrf_saadc_enable();

  NRF_SAADC->TASKS_CALIBRATEOFFSET = 1;
  while ( NRF_SAADC->EVENTS_CALIBRATEDONE == 0 );
  NRF_SAADC->EVENTS_CALIBRATEDONE = 0;
  while ( NRF_SAADC->STATUS == ( SAADC_STATUS_STATUS_Busy << SAADC_STATUS_STATUS_Pos ) );

  NRF_SAADC->TASKS_START = 1;
  NRF_SAADC->TASKS_SAMPLE = 1;
}


void initPPI()
{
  NRF_PPI->CH[PPI_CHANNEL].EEP = ( uint32_t )&NRF_SAADC->EVENTS_END;
  NRF_PPI->CH[PPI_CHANNEL].TEP = ( uint32_t )&NRF_SAADC->TASKS_START;
  NRF_PPI->CHENSET = ( 1UL << PPI_CHANNEL );
}


/**************************
 * Edge Impulse Functions *
 **************************/

/**
 * @brief      Printf function uses vsnprintf and output using Arduino Serial
 *
 * @param[in]  format     Variable argument list
 */
void ei_printf(const char *format, ...) {
    static char print_buf[1024] = { 0 };

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.write(print_buf);
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    free(inference.buffers[0]);
    free(inference.buffers[1]);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif