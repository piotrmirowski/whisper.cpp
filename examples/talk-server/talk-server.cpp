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
    int32_t last_ms    = 1000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t n_chars_hi = 50;
    int32_t max_chars  = 100;

    float vad_thold    = 0.6f;
    float vad_hi_thold = 0.8f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool print_special = false;
    bool print_energy  = false;
    bool no_timestamps = true;

    std::string language    = "en";
    std::string model_wsp   = "models/ggml-base.en.bin";
    std::string url_final   = "http://localhost:8888/speech";
    std::string url_partial = "http://localhost:8888/partialspeech";
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
        else if (arg == "-lms" || arg == "--last-ms")       { params.last_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-nhi" || arg == "--n-chars-hi")    { params.n_chars_hi    = std::stoi(argv[++i]); }
        else if (arg == "-max" || arg == "--max-chars")     { params.max_chars     = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-vhi" || arg == "--vad-hi-thold")  { params.vad_hi_thold  = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"  || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"  || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-ps"  || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-pe"  || arg == "--print-energy")  { params.print_energy  = true; }
        else if (arg == "-l"   || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper") { params.model_wsp     = argv[++i]; }
        else if (arg == "-uf"  || arg == "--url-final")     { params.url_final     = argv[++i]; }
        else if (arg == "-up"  || arg == "--url-partial")   { params.url_partial   = argv[++i]; }
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
    fprintf(stderr, "  -h,       --help            [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N       [%-7d] number of threads to use during computation\n",  params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N      [%-7d] voice duration in milliseconds\n",               params.voice_ms);
    fprintf(stderr, "  -ams N,   --audio-ms N      [%-7d] SDL audio buffer in milliseconds\n",             params.audio_ms);
    fprintf(stderr, "  -dms N,   --detect-ms N     [%-7d] detect part of audio buffer in milliseconds\n",  params.detect_ms);
    fprintf(stderr, "  -lms N,   --last-ms N       [%-7d] last part of audio buffer in milliseconds\n",    params.last_ms);
    fprintf(stderr, "  -c ID,    --capture ID      [%-7d] capture device ID\n",                            params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N    [%-7d] maximum number of tokens per audio chunk\n",     params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N     [%-7d] audio context size (0 - all)\n",                 params.audio_ctx);
    fprintf(stderr, "  -nhi N,   --n-chars-hi N    [%-7d] number of chars when threshold rises to high\n", params.n_chars_hi);
    fprintf(stderr, "  -nax N,   --max-chars N     [%-7d] max number of chars for accepting a sentence\n", params.max_chars);
    fprintf(stderr, "  -vth N,   --vad-thold N     [%-7.2f] final voice activity detection threshold\n",   params.vad_thold);
    fprintf(stderr, "  -vhi N,   --vad-hi-thold N  [%-7.2f] partial voice activity detection threshold\n", params.vad_hi_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N    [%-7.2f] high-pass frequency cutoff\n",                 params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up        [%-7s] speed up audio by x2 (reduced accuracy)\n",      params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate       [%-7s] translate from source language to english\n",    params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special   [%-7s] print special tokens\n",                         params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy    [%-7s] print sound energy (for debugging)\n",           params.print_energy ? "true" : "false");
    fprintf(stderr, "  -l LANG,  --language LANG   [%-7s] spoken language\n",                              params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper   [%-7s] whisper model file\n",                           params.model_wsp.c_str());
    fprintf(stderr, "  -uf URL,  --url-final URL   [%-7s] URL of server for final recognition\n",          params.url_final.c_str());
    fprintf(stderr, "  -up URL,  --url-partial URL [%-7s] URL of server for partial recognition\n",        params.url_partial.c_str());
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

    // Run speech recognition on the audio context.
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    // Extract the text, segment by segment, and compute likelihood.
    int prob_n = 0;
    std::string result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {

        // Extract the text of the current segment.
        const char * text = whisper_full_get_segment_text(ctx, i);
        result += text;

        // Accumulate likelihood of the tokens.
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);
            prob += token.p;
            ++prob_n;
        }
    }
    // Normalise the likelihood of the whole detected sentence.
    if (prob_n > 0) {
        prob /= prob_n;
    }
    // Time the Whisper recognition.
    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}


std::string cleanup_text(const std::string & text_heard) {
    std::string clean_text(text_heard);

    // Remove text between brackets using regex.
    {
        std::regex re("\\[.*?\\]");
        clean_text = std::regex_replace(clean_text, re, "");
    }
    // Remove text between brackets using regex.
    {
        std::regex re("\\(.*?\\)");
        clean_text = std::regex_replace(clean_text, re, "");
    }
    // Remove all characters, except for letters, numbers, punctuation and ':', '\'', '-', ' '.
    clean_text = std::regex_replace(
        clean_text, std::regex("[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), "");
    // Remove leading and trailing whitespace.
    clean_text = std::regex_replace(clean_text, std::regex("^\\s+"), "");
    clean_text = std::regex_replace(clean_text, std::regex("\\s+$"), "");
    // Remove single punctuation.
    clean_text = std::regex_replace(clean_text, std::regex("^[\\.,\\?!\\:\\'\\-]$"), "");

    return clean_text;
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
        // } else {
        //     fprintf(stderr, "%s: Sent %s\n", __func__, request.c_str());
        }
        curl_easy_cleanup(curl);
  }
  return 0;
}


void print_vector_to_file(const std::string & filename, const std::vector<float> & vec) {
    std::ofstream out_file(filename);
    if (!out_file) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }
    for (const auto &elem : vec) {
        out_file << elem << std::endl;
    }
    out_file.close();
}


static inline void clean_sentence(std::string & sentence) {
    sentence = ::trim(sentence);
    if (sentence.rfind("- ") == 0) {
        sentence = sentence.substr(2, sentence.size() - 2);
    }
    sentence = ::trim(sentence);
}


static inline void add_sentence(std::string & sentence,
                                std::vector<std::string> & sentences) {
    clean_sentence(sentence);
    if (sentence.size() > 0) { sentences.push_back(sentence); }
}


std::vector<std::string> split_current_partial(const std::string & text_heard,
                                               std::string & text_carryover) {

    // Replace all occurrences of "--" by "...".
    std::string text = text_heard;
    std::size_t found = text.find("--");
    while (found != std::string::npos) {
        text.replace(found, 2, "...");
        found = text.find("--");
    }

    // Split text into sentences, keep unfinished sentence in text_carryover.
    std::vector<std::string> sentences;
    size_t start = 0;
    while (start < text.size()) {
        size_t finish = text.find_first_of(".!?", start);
        if (finish != std::string::npos) {
            // Search for next non punctuation.
            finish = text.find_first_not_of(".?!", finish);
            finish = (finish != std::string::npos) ? finish : text.size();
            // Extract sentence.
            sentences.push_back(text.substr(start, finish - start));
            start = finish + 1;
        } else {
            if (start == 0) {
                // No punctuation at all in text: consider it a single sentence.
                sentences.push_back(text);
            } else {
                // Keep the unfinished last sentence as text_carryover.
                text_carryover = text.substr(start, text.size() - start);
            }
            start = text.size();
        }
    }

    // Split sentences by speakers ("- ") and clean up.
    std::vector<std::string> sentences_final;
    std::string sentence;
    for (int k = 0; k < sentences.size(); k++) {
        bool new_speaker = (sentences[k].rfind("- ") == 0);
        if (new_speaker && (sentence.size() > 0)) {
            add_sentence(sentence, sentences_final);
            sentence.clear();
        }
        sentence += " " + sentences[k];
    }
    add_sentence(sentence, sentences_final);
    clean_sentence(text_carryover);
    return sentences_final;
 }


 int main(int argc, char ** argv) {

    // Parse parameters and print usage.
    whisper_params params;
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }
    if (whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // Initialise Whisper.
    struct whisper_context * ctx_wsp = whisper_init_from_file(params.model_wsp.c_str());

    // Print some info about the processing.
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

    // Initialise audio.
    audio_async audio(params.audio_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    audio.resume();
    fprintf(stdout, "%s: Initialised Whisper with sample rate %d Hz.\n", __func__, WHISPER_SAMPLE_RATE);

    bool is_running  = true;
    float prob0 = 0.0f;
    int n_pcm_max = WHISPER_SAMPLE_RATE * params.voice_ms / 1000;

    std::vector<float> pcmf32_detect;
    std::vector<float> pcmf32_buff;
    std::vector<float> pcmf32_prev;
    std::string last_text_partial("");
    std::string text_carryover("");
    float vad_thold = params.vad_thold;

    // Main loop.
    while (is_running) {

        // Handle Ctrl + C.
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }
        // Small delay.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        int64_t t_ms = 0;

        {
            // Check last 2s of audio to detect speech.
            audio.get(params.detect_ms, pcmf32_detect);
            bool detected_end = ::vad_simple(pcmf32_detect,
                                             WHISPER_SAMPLE_RATE,
                                             params.last_ms,
                                             vad_thold,
                                             params.freq_thold,
                                             params.print_energy);
            bool detected_pause = ::vad_simple(pcmf32_detect,
                                               WHISPER_SAMPLE_RATE,
                                               params.last_ms,
                                               params.vad_hi_thold,
                                               params.freq_thold,
                                               params.print_energy);
            if (detected_end || detected_pause) {

                // Copy last voice_ms audio context to pcmf32_buff.
                audio.get(params.voice_ms, pcmf32_buff);
                // Prepend any previous audio buffer.
                if (pcmf32_buff.size() >= n_pcm_max) {
                    pcmf32_prev.clear();
                }
                if (pcmf32_prev.size() > 0) {
                    fprintf(stdout, "%s: Prepending audio buffer of %d with previous %d samples.\n",
                            __func__, (int)pcmf32_buff.size(), (int)pcmf32_prev.size());
                    pcmf32_buff.insert(pcmf32_buff.begin(), pcmf32_prev.begin(), pcmf32_prev.end());
                }

                // Transcribe audio to text_heard using Whisper, then clean up results.
                std::string text_heard = ::trim(::transcribe(ctx_wsp, params, pcmf32_buff, prob0, t_ms));
                text_heard = ::cleanup_text(text_heard);

                // Skip empty lines or verbose the result.
                if (text_heard.empty()) {
                    fprintf(stdout, "%s: Heard nothing, skipping... (t = %d ms)\n", __func__, (int)t_ms);
                    continue;
                }
                if (text_heard == last_text_partial) {
                    continue;
                }
                fprintf(stdout, "\n%s: Detected speech! (t = %d ms)\n", __func__, (int)t_ms);

                // Partial detection: start loosening the detection threshold?
                if (!detected_end) {
                    vad_thold += (params.vad_hi_thold - vad_thold) / 2;
                    fprintf(stdout, "%s: Raising vad_thold to %f\n", __func__, vad_thold);
                }

                // End if the text is too long.
                if (text_heard.size() > params.max_chars) {
                    fprintf(stdout, "%s: Cutting after %d chars\n", __func__, (int)text_heard.size());
                }
                detected_end = detected_end || (text_heard.size() > params.max_chars);

                if (detected_end) {

                    // Append the carry over to the final sentence.
                    fprintf(stdout, "%s: Final %s%s...%s %s%s%s\n", __func__,
                        "\033[1;32m", text_carryover.c_str(), "\033[0m",
                        "\033[1;31m", text_heard.c_str(), "\033[0m");
                    text_heard = text_carryover + "... " + text_heard;

                    // Send the line to server as final recognition.
                    text_carryover.clear();
                    std::vector<std::string> sentences = split_current_partial(text_heard, text_carryover);
                    for (int k = 0; k < sentences.size(); k++) {
                        post_text(sentences[k], params.url_final);
                    }

                    // Reset the last partially detected text.
                    last_text_partial = "";
                    vad_thold = params.vad_thold;

                    // Reset audio context, keeping audio buffer what has not been processed yet.
                    int n_final = pcmf32_buff.size();
                    audio.get(params.voice_ms, pcmf32_buff);
                    audio.clear();
                    int n_buff = pcmf32_buff.size();
                    int n_copy = n_buff - n_final;
                    if ((n_final < n_pcm_max) && (n_buff < n_pcm_max) && (n_copy > 0)) {
                        pcmf32_prev.resize(n_copy);
                        std::copy(pcmf32_buff.end() - n_copy, pcmf32_buff.end(), pcmf32_prev.begin());
                        fprintf(stdout, "%s: Kept buffer of %d audio samples\n", __func__, n_copy);
                    } else {
                        pcmf32_prev.clear();
                    }

                } else {
                    fprintf(stdout, "%s: Partial: %s%s%s\n", __func__, "\033[1;34m", text_heard.c_str(), "\033[0m");

                    // Send the line to server as partial recognition.
                    post_text(text_heard, params.url_partial);

                    // Store the last partially detected text.
                    last_text_partial = text_heard;
                }
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
