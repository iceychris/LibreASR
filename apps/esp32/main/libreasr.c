#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "board.h"
#include "audio_common.h"
#include "audio_pipeline.h"
#include "i2s_stream.h"
#include "raw_stream.h"
#include "filter_resample.h"

// WiFi
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event_loop.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "lwip/err.h"
#include "lwip/sys.h"

// GPIO
#include "driver/gpio.h"

// websockets
#include "esp_websocket_client.h"


#define xstr(s) str(s) 
#define str(s) #s

/* The examples use WiFi configuration that you can set via project configuration menu
   If you'd rather not, just change the below entries to strings with
   the config you want - ie #define EXAMPLE_WIFI_SSID "mywifissid"
*/
#define ESP_WIFI_SSID      "DEADBEEF"
#define ESP_WIFI_PASS      "deadbeef314"
#define ESP_MAXIMUM_RETRY  3

/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;

/* The event group allows multiple bits for each event, but we only care about two events:
 * - we are connected to the AP with an IP
 * - we failed to connect after the maximum amount of retries */
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static int s_retry_num = 0;


#define LIBREASR_URI "ws://libreasr/asupersecretwebsocketpath345"
#define LIBREASR_PORT 8080
esp_websocket_client_handle_t ws_client;


static const char *TAG = "ASR";

#define VAD_SAMPLE_RATE_HZ 16000
#define VAD_FRAME_LENGTH_MS 80
#define VAD_BUFFER_LENGTH (VAD_FRAME_LENGTH_MS * VAD_SAMPLE_RATE_HZ / 1000)
#define FLOAT_DATA_LENGTH (VAD_BUFFER_LENGTH + 2)
static float * float_data = NULL;

/* leds */
#define GPIO_OUTPUT_IO_0    22
#define GPIO_OUTPUT_PIN_SEL  ((1ULL<<GPIO_OUTPUT_IO_0))
#define ESP_INTR_FLAG_DEFAULT 0
static xQueueHandle leds_evt_queue = NULL;

/* language switching */
static char * text = NULL;
static int text_offset = 0;
static char * lang = "en  ";
#define TEXT_LEN 1024


static void reset_text() {
    text_offset = 0;
    if (text != NULL) {
        memset(text, 0, TEXT_LEN);
    } else {
        text = (char *) malloc(TEXT_LEN * sizeof(char));
        if (text == NULL) {
            ESP_LOGE(TAG, "Memory allocation (text) failed!");
        }
    }
}


// forward declaration
static void websocket_app_start(void);


static uint32_t maybe_switch_lang(char * transcript, int len) {
    uint32_t blinks = len;

    // add to text
    for (int i = 0; i < len; i++) {
	text[text_offset + i] = transcript[i];
    }
    text[text_offset + len] = '\0';
    text_offset += len;
    ESP_LOGI(TAG, "Text: %s", text);

    // switching logic
    if (strcmp(lang, "en  ") == 0) {
	if ((strstr(text, "switch") != NULL || strstr(text, "change") != NULL) && (strstr(text, "ger") != NULL || strstr(text, "geo") != NULL)) {
            ESP_LOGI(TAG, "! ! !");
            ESP_LOGI(TAG, "switching to de...");
            ESP_LOGI(TAG, "! ! !");
	    lang = "de  ";
	    blinks += 20;
	    reset_text();
	    websocket_app_start();
	}
    }
    if (strcmp(lang, "de  ") == 0) {
	if (strstr(text, "wechsel") != NULL && strstr(text, "eng") != NULL) {
            ESP_LOGI(TAG, "! ! !");
            ESP_LOGI(TAG, "switching to en...");
            ESP_LOGI(TAG, "! ! !");
	    lang = "en  ";
	    blinks += 20;
	    reset_text();
	    websocket_app_start();
	}
    }
    return blinks;
}


static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
        ESP_LOGI(TAG, "WEBSOCKET_EVENT_CONNECTED");
        break;
    case WEBSOCKET_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "WEBSOCKET_EVENT_DISCONNECTED");
        break;
    case WEBSOCKET_EVENT_DATA:
        if (data->op_code == 9 || data->op_code == 10) break;
        // ESP_LOGI(TAG, "WEBSOCKET_EVENT_DATA");
        // ESP_LOGI(TAG, "Received opcode=%d", data->op_code);
        // ESP_LOGW(TAG, "Received=%.*s", data->data_len, (char *)data->data_ptr);
        // ESP_LOGW(TAG, "Total payload length=%d, data_len=%d, current payload offset=%d\r\n", data->payload_len, data->data_len, data->payload_offset);
	uint32_t blinks = maybe_switch_lang((char *) data->data_ptr, data->data_len);
        xQueueSendToBack(leds_evt_queue, &blinks, NULL);
        break;
    case WEBSOCKET_EVENT_ERROR:
        ESP_LOGI(TAG, "WEBSOCKET_EVENT_ERROR");
        break;
    }
}


static esp_err_t event_handler(void *ctx, system_event_t *event)
{
    switch(event->event_id) {
    case SYSTEM_EVENT_STA_START:
        esp_wifi_connect();
        break;
    case SYSTEM_EVENT_STA_GOT_IP:
        ESP_LOGI(TAG, "got ip:%s",
                 ip4addr_ntoa(&event->event_info.got_ip.ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        break;
    case SYSTEM_EVENT_STA_DISCONNECTED:
        {
            if (s_retry_num < ESP_MAXIMUM_RETRY) {
                esp_wifi_connect();
                xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
                s_retry_num++;
                ESP_LOGI(TAG,"retry to connect to the AP");
            }
            ESP_LOGI(TAG,"failed to connect to the AP! Are the WiFi-Credentials okay?\n");
            break;
        }
    default:
        break;
    }
    return ESP_OK;
}


void wifi_init_sta()
{
    s_wifi_event_group = xEventGroupCreate();

    tcpip_adapter_init();
    ESP_ERROR_CHECK(esp_event_loop_init(event_handler, NULL) );

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = ESP_WIFI_SSID,
            .password = ESP_WIFI_PASS,
            /* Setting a password implies station will connect to all security modes including WEP/WPA.
             * However these modes are deprecated and not advisable to be used. Incase your Access point
             * doesn't support WPA2, these mode can be enabled by commenting below line */
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA) );
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config) );
    ESP_ERROR_CHECK(esp_wifi_start() );

    ESP_LOGI(TAG, "wifi_init_sta finished.");
    ESP_LOGI(TAG, "connect to ap SSID:%s password:%s",
             ESP_WIFI_SSID, ESP_WIFI_PASS);
}


static void websocket_app_start(void)
{
    esp_websocket_client_config_t websocket_cfg = {};
    websocket_cfg.uri = LIBREASR_URI;
    websocket_cfg.port = LIBREASR_PORT;

    ESP_LOGI(TAG, "Connecting to %s...", websocket_cfg.uri);

    ws_client = esp_websocket_client_init(&websocket_cfg);
    esp_websocket_register_events(ws_client, WEBSOCKET_EVENT_ANY, websocket_event_handler, (void *)ws_client);

    esp_websocket_client_start(ws_client);

    // wait for ws connection
    ESP_LOGI(TAG, "Waiting for WS to connect...");
    while (!esp_websocket_client_is_connected(ws_client)) {
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
    ESP_LOGI(TAG, "WS connected.");
}


// buf: buffer of samples
// len: number of shorts in that buffer
static void send_voice_data(int16_t * buf, size_t len) {

    int offset = 0;

    // store lang
    uint8_t lang_arr[4];
    for (int i = 0; i < 4; i++) {
        lang_arr[i] = (uint8_t) lang[i];
    }
    float_data[offset] = *((float *) lang_arr);
    offset += 1;

    // store sr
    float_data[offset] = (float) VAD_SAMPLE_RATE_HZ;
    offset += 1;

    // store data
    // convert short to float
    for (int i = 0; i < len; i++) {
        float_data[i + offset] = ((float) buf[i]) * 0.0002;
    }

    // send
    if (esp_websocket_client_is_connected(ws_client)) {

        // ESP_LOGI(TAG, "Sending %d bytes over websocket", binn_size(obj));
        esp_websocket_client_send_bin(ws_client, (char *) float_data, FLOAT_DATA_LENGTH * 4, 250 / portTICK_RATE_MS);

        // yield
        // vTaskDelay(1 / portTICK_PERIOD_MS);
    }
}


static void leds_task(void* arg)
{
    uint32_t len;
    for(;;) {
        if(xQueueReceive(leds_evt_queue, &len, portMAX_DELAY)) {
            // printf("LEDs: received payload of size %d\n", len);
            for (int i = 0; i < len; i++) {
                gpio_set_level(GPIO_OUTPUT_IO_0, i % 2);
                vTaskDelay(50 / portTICK_RATE_MS);
            }
        }
    }
}


static void drive_leds_start() {
    gpio_config_t io_conf;
    //disable interrupt
    io_conf.intr_type = GPIO_INTR_DISABLE;
    //set as output mode
    io_conf.mode = GPIO_MODE_OUTPUT;
    //bit mask of the pins that you want to set,e.g.GPIO18/19
    io_conf.pin_bit_mask = GPIO_OUTPUT_PIN_SEL;
    //disable pull-down mode
    io_conf.pull_down_en = 0;
    //disable pull-up mode
    io_conf.pull_up_en = 0;
    //configure GPIO with the given settings
    gpio_config(&io_conf);

    //create a queue to handle gpio event
    leds_evt_queue = xQueueCreate(10, sizeof(uint32_t));
    //start task
    xTaskCreate(leds_task, "leds_task", 2048, NULL, 10, NULL);
}


void app_main()
{
    // enable leds
    drive_leds_start();


    ///
    // connect to WiFi
    ///

    //Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "ESP_WIFI_MODE_STA");
    wifi_init_sta();


    ///
    // connect to websocket
    ///

    websocket_app_start();


    ///
    // audio stuff
    ///

    esp_log_level_set("*", ESP_LOG_WARN);
    esp_log_level_set(TAG, ESP_LOG_INFO);

    audio_pipeline_handle_t pipeline;
    audio_element_handle_t i2s_stream_reader, filter, raw_read;

    ESP_LOGI(TAG, "[ 1 ] Start codec chip");
    audio_board_handle_t board_handle = audio_board_init();
    audio_hal_ctrl_codec(board_handle->audio_hal, AUDIO_HAL_CODEC_MODE_BOTH, AUDIO_HAL_CTRL_START);

    ESP_LOGI(TAG, "[ 2 ] Create audio pipeline for recording");
    audio_pipeline_cfg_t pipeline_cfg = DEFAULT_AUDIO_PIPELINE_CONFIG();
    pipeline = audio_pipeline_init(&pipeline_cfg);
    mem_assert(pipeline);

    ESP_LOGI(TAG, "[2.1] Create i2s stream to read audio data from codec chip");
    i2s_stream_cfg_t i2s_cfg = I2S_STREAM_CFG_DEFAULT();
    i2s_cfg.i2s_config.sample_rate = 48000;
    i2s_cfg.type = AUDIO_STREAM_READER;
#if defined CONFIG_ESP_LYRAT_MINI_V1_1_BOARD
    i2s_cfg.i2s_port = 1;
#endif
    i2s_stream_reader = i2s_stream_init(&i2s_cfg);

    ESP_LOGI(TAG, "[2.2] Create filter to resample audio data");
    rsp_filter_cfg_t rsp_cfg = DEFAULT_RESAMPLE_FILTER_CONFIG();
    rsp_cfg.src_rate = 48000;
    rsp_cfg.src_ch = 2;
    rsp_cfg.dest_rate = VAD_SAMPLE_RATE_HZ;
    rsp_cfg.dest_ch = 1;
    filter = rsp_filter_init(&rsp_cfg);

    ESP_LOGI(TAG, "[2.3] Create raw to receive data");
    raw_stream_cfg_t raw_cfg = {
        .out_rb_size = 8 * 1024,
        .type = AUDIO_STREAM_READER,
    };
    raw_read = raw_stream_init(&raw_cfg);

    ESP_LOGI(TAG, "[ 3 ] Register all elements to audio pipeline");
    audio_pipeline_register(pipeline, i2s_stream_reader, "i2s");
    audio_pipeline_register(pipeline, filter, "filter");
    audio_pipeline_register(pipeline, raw_read, "raw");

    ESP_LOGI(TAG, "[ 4 ] Link elements together [codec_chip]-->i2s_stream-->filter-->raw-->[VAD]");
    const char *link_tag[3] = {"i2s", "filter", "raw"};
    audio_pipeline_link(pipeline, &link_tag[0], 3);

    ESP_LOGI(TAG, "[ 5 ] Start audio_pipeline");
    audio_pipeline_run(pipeline);

    int16_t *vad_buff = (int16_t *)malloc(VAD_BUFFER_LENGTH * sizeof(short));
    if (vad_buff == NULL) {
        ESP_LOGE(TAG, "Memory allocation (vad_buff) failed!");
        goto abort_speech_detection;
    }

    // allocate
    float_data = (float *)malloc(FLOAT_DATA_LENGTH * sizeof(float));
    if (float_data == NULL) {
        ESP_LOGE(TAG, "Memory allocation (float_data) failed!");
        goto abort_speech_detection;
    }
    reset_text();

    while (1) {
        raw_stream_read(raw_read, (char *)vad_buff, VAD_BUFFER_LENGTH * sizeof(short));

        // send
        send_voice_data(vad_buff, VAD_BUFFER_LENGTH);
    }

    free(vad_buff);
    vad_buff = NULL;

abort_speech_detection:

    ESP_LOGI(TAG, "[ 8 ] Stop audio_pipeline and release all resources");
    audio_pipeline_stop(pipeline);
    audio_pipeline_wait_for_stop(pipeline);
    audio_pipeline_terminate(pipeline);

    /* Terminate the pipeline before removing the listener */
    audio_pipeline_remove_listener(pipeline);

    audio_pipeline_unregister(pipeline, i2s_stream_reader);
    audio_pipeline_unregister(pipeline, filter);
    audio_pipeline_unregister(pipeline, raw_read);

    /* Release all resources */
    audio_pipeline_deinit(pipeline);
    audio_element_deinit(i2s_stream_reader);
    audio_element_deinit(filter);
    audio_element_deinit(raw_read);

}
