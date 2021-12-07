#include <Arduino.h>

#define RED_LED 22
#define GREEN_LED 23

#define CLASSIFICATION_CUTOFF 0.70  // Threshold to consider positive detection of asthma event.
#define HOLD_TIME 4     // How long to let timer run while counting number of asthma attacks. [Seconds]
                        // Max is given by 2^16 / 2^12 (2^Timer Bitwidth / 2^Timer Prescaler)
#define NOTIFY_AMOUNT 3 // How many asthma occurences required to trigger notification.

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 3

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <dog-asthma-detection_inferencing.h>
#include <stdio.h>
#include "mbed.h"
#include "rtos.h"

/** Define Threads */
//rtos::Thread threadClassify;
rtos::Thread threadRedLED(osPriorityAboveNormal);
rtos::Thread threadNotify(osPriorityAboveNormal1);

/** Timer Functions & counter */
void initTimer4();
volatile bool timer_running = false;
volatile int asthma_count = 0;

/**Define Tasks */
//void Classify();
void RedLED();
void Notify();

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
    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    digitalWrite(RED_LED, HIGH); //High is off for onboard LEDs
    digitalWrite(GREEN_LED, HIGH);

    // put your setup code here, to run once:
    Serial.begin(115200);

    Serial.println("Edge Impulse Inferencing Demo");

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                            sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Failed to setup audio sampling\r\n");
        return;
    }

    // Init Timer4 and PPI
    initTimer4();

    //threadClassify.start(Classify);
    threadRedLED.start(RedLED);
    threadNotify.start(Notify);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
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
}

// ISR for Timer4
extern "C" void TIMER4_IRQHandler_v( void )
{
  //stopTimer4();
    if (NRF_TIMER4->EVENTS_COMPARE[0] == 1)
    {   
        timer_running = false;
        asthma_count = 0;
        NRF_TIMER4->TASKS_STOP = 1;
        NRF_TIMER4->TASKS_CLEAR = 1;
        NRF_TIMER4->EVENTS_COMPARE[0] = 0;
    }
}

void RedLED () {
  while(1) {
    rtos::ThisThread::flags_wait_any(0x1);
    digitalWrite(RED_LED, LOW);
    rtos::ThisThread::sleep_for(50); // wait
    digitalWrite(RED_LED, HIGH);
    rtos::ThisThread::sleep_for(50); // wait
  }
}

void Notify () {
  while(1) {
    rtos::ThisThread::flags_wait_any(0x1);
    digitalWrite(GREEN_LED, LOW);
    rtos::ThisThread::sleep_for(50); // wait
    digitalWrite(GREEN_LED, HIGH);
    rtos::ThisThread::sleep_for(50); // wait
  }
}

void initTimer4()
{
  NRF_TIMER4->MODE = TIMER_MODE_MODE_Timer;
  NRF_TIMER4->TASKS_CLEAR = 1;
  NRF_TIMER4->BITMODE = TIMER_BITMODE_BITMODE_16Bit;
  NRF_TIMER4->SHORTS = TIMER_SHORTS_COMPARE0_CLEAR_Enabled << TIMER_SHORTS_COMPARE0_CLEAR_Pos;
  NRF_TIMER4->PRESCALER = 10;
  NRF_TIMER4->CC[0] = 15625 * HOLD_TIME; // Needs prescaler set to 10 (15625 = 16MHz / 2^10) Cant be larger than 2^16

  NRF_TIMER4->INTENSET = TIMER_INTENSET_COMPARE0_Enabled << TIMER_INTENSET_COMPARE0_Pos;
  //NVIC_SetPriority( TIMER4_IRQn, 1UL );
  NVIC_EnableIRQ( TIMER4_IRQn );
}


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
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
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

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

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
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif