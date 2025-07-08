#ifndef CONFIG_HPP
#define CONFIG_HPP
namespace Config {
    // Heltallskonstanter
    inline constexpr int MAX_FRAMES = 200;
    inline constexpr int WINDOW_WIDTH = 300;
    inline constexpr int WINDOW_HEIGHT = 300;

    //Kamera
    inline constexpr const char* VIDEO_OUT ="/mnt/c/Self-Study/C++/Dataset_Real/vid_1.mp4";
    inline constexpr int FPS = 30;

    //IA-modell readNetFromCaffe
    inline constexpr const char* DEPLOY_MODEL = "/mnt/c/Self-Study/C++/deploy.prototxt";
    inline constexpr const char* CAFFE_MODEL = "/mnt/c/Self-Study/C++/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    //EVM
    inline constexpr int LEVELS = 1;
    inline constexpr int ALPHA = 100;
    inline constexpr float LOW_HEART = 0.83; // Lav grense for hjertefrekvens i Hz
    inline constexpr float HIGH_HEART = 3.0; // Høy grense for hjertefrekvens i Hz
    inline constexpr float LOW_RESP = 0.18;  // Lav grense for respirasjonsfrekvens i Hz
    inline constexpr float HIGH_RESP = 0.5;  // Høy grense for respirasjonsfrekvens i Hz
}
#endif