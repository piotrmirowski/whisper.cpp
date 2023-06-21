// Talk with a server.
// Modified by Piotr Mirowski (piotr.mirowski@computer.org) from ggerganov/whisper.cpp/examples/talk

#include "common.h"
#include "common-sdl.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>


// Command-line parameters.
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 30000;
    int32_t audio_ms   = 60000;
    int32_t detect_ms  = 2000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool print_special = false;
    bool print_energy  = false;
    bool no_timestamps = true;

    std::string language  = "en";
    std::string model_wsp = "models/ggml-base.en.bin";
    std::string url_server = "http://localhost:8888/speech";
};


void whisper_print_usage(int argc, char ** argv, const whisper_params & params);


bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (arg == "-vms" || arg == "--voice-ms")      { params.voice_ms      = std::stoi(argv[++i]); }
        else if (arg == "-ams" || arg == "--audio-ms")      { params.audio_ms      = std::stoi(argv[++i]); }
        else if (arg == "-dms" || arg == "--detect-ms")     { params.detect_ms     = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-pe"  || arg == "--print-energy")  { params.print_energy  = true; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper") { params.model_wsp     = argv[++i]; }
        else if (arg == "-u"   || arg == "--url_server")    { params.url_server    = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}


void whisper_print_usage(int argc, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -ams N,   --audio-ms N     [%-7d] SDL audio buffer in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up       [%-7s] speed up audio by x2 (reduced accuracy)\n",     params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -u URL,   --url-server URL [%-7s] URL of server\n",                               params.url_server.c_str());
    fprintf(stderr, "\n");
}


std::string transcribe(whisper_context * ctx,
                       const whisper_params & params,
                       const std::vector<float> & pcmf32,
                       float & prob,
                       int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.audio_ctx        = params.audio_ctx;
    wparams.speed_up         = params.speed_up;

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}


int post_text(const std::string & text, const std::string & url_server) {
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();

    if (curl) {
        // Assemble POST request.
        curl_easy_setopt(curl, CURLOPT_URL, url_server.c_str());
        std::string request("text=");
        request += text;
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long) request.size());
        // Perform the request (res will get the return code) and check for errors.
        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stderr, "%s: curl_easy_perform() failed on %s: %s\n",
                    __func__, url_server.c_str(), curl_easy_strerror(res));
        } else {
            fprintf(stderr, "%s: Sent %s\n", __func__, request.c_str());
        }
        curl_easy_cleanup(curl);
  }
  return 0;
}


int main(int argc, char ** argv) {
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // whisper init

    struct whisper_context * ctx_wsp = whisper_init_from_file(params.model_wsp.c_str());

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }


    // Init audio.
    audio_async audio(params.audio_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    audio.resume();
    fprintf(stdout, "%s: Initialised Whisper with sample rate %d.\n", __func__, WHISPER_SAMPLE_RATE);

    bool is_running  = true;
    float prob0 = 0.0f;

    std::vector<float> pcmf32_cur;

    // Main loop.
    while (is_running) {

        // Handle Ctrl + C.
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }
        // Small delay.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        int64_t t_ms = 0;

        {
            // Check last 2s of audio to detect speech.
            audio.get(params.detect_ms, pcmf32_cur);
            if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1250, params.vad_thold, params.freq_thold, params.print_energy)) {
                fprintf(stdout, "%s: Speech detected! Transcribing...\n", __func__);

                // Copy last voice_ms audio context and clear audio buffer.
                audio.get(params.voice_ms, pcmf32_cur);
                audio.clear();

                // Transcribe audio to text_heard using Whisper.
                std::string text_heard;
                text_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_cur, prob0, t_ms));
                fprintf(stdout, "%s: Transcribed %d frames.\n", __func__, (int) pcmf32_cur.size());

                // Remove text between brackets using regex.
                {
                    std::regex re("\\[.*?\\]");
                    text_heard = std::regex_replace(text_heard, re, "");
                }
                // Remove text between brackets using regex.
                {
                    std::regex re("\\(.*?\\)");
                    text_heard = std::regex_replace(text_heard, re, "");
                }
                // Remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '.
                text_heard = std::regex_replace(text_heard, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), "");
                // Take first line (drop this one)
                // text_heard = text_heard.substr(0, text_heard.find_first_of('\n'));
                // Remove leading and trailing whitespace.
                text_heard = std::regex_replace(text_heard, std::regex("^\\s+"), "");
                text_heard = std::regex_replace(text_heard, std::regex("\\s+$"), "");

                // Skip empty lines or verbose the result.
                if (text_heard.empty()) {
                    fprintf(stdout, "%s: Heard nothing, skipping... (t = %d ms)\n", __func__, (int) t_ms);
                    continue;
                }
                fprintf(stdout, "%s: Heard '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", text_heard.c_str(), "\033[0m", (int) t_ms);

                // Send the line to server.
                post_text(text_heard, params.url_server);
            }
        }
    }

    // Exit audio and Whisper context gracefully.
    fprintf(stdout, "%s: Pausing audio and deleting Whisper context.\n", __func__);
    audio.pause();
    whisper_print_timings(ctx_wsp);
    whisper_free(ctx_wsp);

    return 0;
}
