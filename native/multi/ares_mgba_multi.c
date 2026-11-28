/* Multi-console (GB/GBC/GBA) instance-owned mGBA 0.10.5 interface for the ARES-U game
 * abstraction layer. SPDX-License-Identifier: MPL-2.0 */
#include <mgba/core/core.h>
#include <mgba/core/config.h>
#include <mgba/core/log.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define ARES_API __declspec(dllexport)
#else
#define ARES_API __attribute__((visibility("default")))
#endif
#define ARES_TLS _Thread_local

#define MAX_W 256
#define MAX_H 224

typedef struct {
    struct mCore* core;
    unsigned width, height;
    color_t pixels[MAX_W * MAX_H];
} AresCore;

static ARES_TLS char ares_error[256];

static void ares_log(struct mLogger* logger, int category, enum mLogLevel level, const char* format, va_list args) {
    (void) logger;
    (void) category;
    if (level & (mLOG_FATAL | mLOG_ERROR)) {
        vfprintf(stderr, format, args);
        fputc('\n', stderr);
    }
}
static struct mLogger ares_logger = { ares_log, NULL };

static void set_error(const char* message) {
    strncpy(ares_error, message, sizeof(ares_error) - 1);
    ares_error[sizeof(ares_error) - 1] = 0;
}

ARES_API const char* ares_last_error(void) { return ares_error; }

ARES_API void* ares_create(const char* rom) {
    mLogSetDefaultLogger(&ares_logger);
    if (!rom || sizeof(color_t) != 4) {
        set_error("ROM path required and mGBA must use 32-bit color");
        return NULL;
    }
    AresCore* h = calloc(1, sizeof(*h));
    if (!h) { set_error("Unable to allocate core handle"); return NULL; }
    h->core = mCoreFind(rom);
    if (!h->core) { free(h); set_error("No mGBA core for this ROM"); return NULL; }
    if (!h->core->init(h->core)) {
        free(h->core); free(h); set_error("Core initialization failed"); return NULL;
    }
    mCoreInitConfig(h->core, "ares");
    h->core->opts.useBios = false;
    h->core->opts.skipBios = true;
    h->core->opts.mute = true;
    h->core->opts.volume = 0;
    mCoreConfigSetIntValue(&h->core->config, "sgb.borders", 0);
    h->core->loadConfig(h->core, &h->core->config);
    h->core->setVideoBuffer(h->core, h->pixels, MAX_W);
    if (!mCoreLoadFile(h->core, rom)) {
        mCoreConfigDeinit(&h->core->config);
        h->core->deinit(h->core);
        free(h);
        set_error("ROM loading failed");
        return NULL;
    }
    h->core->reset(h->core);
    h->core->desiredVideoDimensions(h->core, &h->width, &h->height);
    if (h->width > MAX_W || h->height > MAX_H) {
        mCoreConfigDeinit(&h->core->config);
        h->core->deinit(h->core);
        free(h);
        set_error("Unexpected video dimensions");
        return NULL;
    }
    ares_error[0] = 0;
    return h;
}

ARES_API void ares_destroy(void* ptr) {
    if (!ptr) return;
    AresCore* h = ptr;
    mCoreConfigDeinit(&h->core->config);
    h->core->deinit(h->core); /* deinit owns/frees mCore */
    free(h);
}

/* enum mPlatform: 0 = GBA, 1 = GB/GBC. */
ARES_API int ares_platform(void* ptr) {
    if (!ptr) return -1;
    AresCore* h = ptr;
    return (int) h->core->platform(h->core);
}

ARES_API int ares_video_size(void* ptr, unsigned* width, unsigned* height) {
    if (!ptr || !width || !height) { set_error("Invalid video size query"); return 0; }
    AresCore* h = ptr;
    h->core->desiredVideoDimensions(h->core, &h->width, &h->height);
    *width = h->width;
    *height = h->height;
    return 1;
}

/* Row stride of ares_pixels() in pixels. */
ARES_API unsigned ares_stride(void* ptr) { (void) ptr; return MAX_W; }

ARES_API void ares_step(void* ptr) {
    if (ptr) { AresCore* h = ptr; h->core->runFrame(h->core); }
}

ARES_API void ares_set_keys(void* ptr, uint32_t keys) {
    if (ptr) { AresCore* h = ptr; h->core->setKeys(h->core, keys & 0x3FF); }
}

static int valid_span(AresCore* h, uint32_t address, size_t size) {
    uint64_t end = (uint64_t) address + size;
    if (h->core->platform(h->core) == mPLATFORM_GB) return end <= 0x10000;
    if (end > UINT32_MAX || size > 0x02000000) return 0;
    if (address >= 0x02000000 && end <= 0x02040000) return 1;
    if (address >= 0x03000000 && end <= 0x03008000) return 1;
    if (address >= 0x08000000 && end <= 0x0A000000) return 1;
    return 0;
}

ARES_API int ares_read(void* ptr, uint32_t address, void* output, size_t size) {
    if (!ptr || (!output && size) || !valid_span(ptr, address, size)) { set_error("Invalid read span"); return 0; }
    AresCore* h = ptr;
    uint8_t* out = output;
    for (size_t i = 0; i < size; ++i) out[i] = (uint8_t) h->core->busRead8(h->core, address + (uint32_t) i);
    return 1;
}

/* Banked read (e.g. GBC WRAM bank ``segment`` at 0xD000-0xDFFF), independent of the bank mapped now. */
ARES_API int ares_read_raw(void* ptr, uint32_t address, int segment, void* output, size_t size) {
    if (!ptr || (!output && size) || !valid_span(ptr, address, size)) { set_error("Invalid raw read span"); return 0; }
    AresCore* h = ptr;
    uint8_t* out = output;
    for (size_t i = 0; i < size; ++i) out[i] = (uint8_t) h->core->rawRead8(h->core, address + (uint32_t) i, segment);
    return 1;
}

ARES_API size_t ares_state_size(void* ptr) {
    if (!ptr) return 0;
    AresCore* h = ptr;
    return h->core->stateSize(h->core);
}

ARES_API int ares_save_state(void* ptr, void* output, size_t size) {
    if (!ptr || !output || size != ares_state_size(ptr)) { set_error("Invalid save state buffer"); return 0; }
    AresCore* h = ptr;
    return h->core->saveState(h->core, output);
}

ARES_API int ares_load_state(void* ptr, const void* input, size_t size) {
    if (!ptr || !input || size != ares_state_size(ptr)) { set_error("Invalid load state buffer"); return 0; }
    AresCore* h = ptr;
    return h->core->loadState(h->core, input);
}

ARES_API const uint32_t* ares_pixels(void* ptr) {
    return ptr ? (const uint32_t*) ((AresCore*) ptr)->pixels : NULL;
}
