/* Opaque, instance-owned mGBA 0.10.5 interface. SPDX-License-Identifier: MPL-2.0 */
#include <mgba/core/core.h>
#include <mgba/core/config.h>
#include <mgba/core/log.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define ARES_API __declspec(dllexport)
#ifdef _MSC_VER
#define ARES_TLS __declspec(thread)
#else
#define ARES_TLS _Thread_local
#endif
#else
#define ARES_API __attribute__((visibility("default")))
#define ARES_TLS _Thread_local
#endif

typedef struct {
    struct mCore* core;
    color_t pixels[240 * 160];
} AresCore;
static ARES_TLS char ares_error[256];
static void ares_log(struct mLogger* logger, int category, enum mLogLevel level,
                     const char* format, va_list args) {
    (void) logger;
    (void) category;
    if (level & (mLOG_FATAL | mLOG_ERROR | mLOG_WARN)) {
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
    h->core = mCoreCreate(mPLATFORM_GBA);
    if (!h->core) { free(h); set_error("GBA core unavailable"); return NULL; }
    if (!h->core->init(h->core)) {
        free(h->core); free(h); set_error("Core initialization failed"); return NULL;
    }
    mCoreInitConfig(h->core, "ares");
    h->core->opts.useBios = false;
    h->core->opts.skipBios = true;
    h->core->opts.mute = true;
    h->core->opts.volume = 0;
    h->core->loadConfig(h->core, &h->core->config);
    h->core->setVideoBuffer(h->core, h->pixels, 240);
    if (!mCoreLoadFile(h->core, rom)) {
        mCoreConfigDeinit(&h->core->config);
        h->core->deinit(h->core); free(h);
        set_error("ROM loading failed"); return NULL;
    }
    h->core->reset(h->core);
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
ARES_API void ares_step(void* ptr) {
    if (ptr) { AresCore* h = ptr; h->core->runFrame(h->core); }
}
ARES_API void ares_set_keys(void* ptr, uint32_t keys) {
    if (ptr) { AresCore* h = ptr; h->core->setKeys(h->core, keys & 0x3FF); }
}
static int valid_span(uint32_t address, size_t size, int writing) {
    uint64_t end = (uint64_t) address + size;
    if (end > UINT32_MAX || size > 0x02000000) return 0;
    if (address >= 0x02000000 && end <= 0x02040000) return 1;
    if (address >= 0x03000000 && end <= 0x03008000) return 1;
    if (!writing && address >= 0x08000000 && end <= 0x0A000000) return 1;
    return 0;
}
ARES_API int ares_read(void* ptr, uint32_t address, void* output, size_t size) {
    if (!ptr || (!output && size) || !valid_span(address, size, 0)) {
        set_error("Invalid read span"); return 0;
    }
    AresCore* h = ptr;
    uint8_t* out = output;
    for (size_t i = 0; i < size; ++i) out[i] = (uint8_t) h->core->busRead8(h->core, address + (uint32_t) i);
    return 1;
}
ARES_API int ares_write(void* ptr, uint32_t address, const void* input, size_t size) {
    if (!ptr || (!input && size) || !valid_span(address, size, 1)) {
        set_error("Invalid write span"); return 0;
    }
    AresCore* h = ptr;
    const uint8_t* in = input;
    for (size_t i = 0; i < size; ++i) h->core->busWrite8(h->core, address + (uint32_t) i, in[i]);
    return 1;
}
ARES_API size_t ares_state_size(void* ptr) {
    if (!ptr) return 0;
    AresCore* h = ptr;
    return h->core->stateSize(h->core);
}
ARES_API int ares_save_state(void* ptr, void* output, size_t size) {
    if (!ptr || !output || size != ares_state_size(ptr)) {
        set_error("Invalid save state buffer"); return 0;
    }
    AresCore* h = ptr;
    return h->core->saveState(h->core, output);
}
ARES_API int ares_load_state(void* ptr, const void* input, size_t size) {
    if (!ptr || !input || size != ares_state_size(ptr)) {
        set_error("Invalid load state buffer"); return 0;
    }
    AresCore* h = ptr;
    return h->core->loadState(h->core, input);
}
ARES_API const uint32_t* ares_pixels(void* ptr) {
    return ptr ? (const uint32_t*) ((AresCore*) ptr)->pixels : NULL;
}
ARES_API size_t ares_battery_size(void* ptr) {
    if (!ptr) return 0;
    AresCore* h = ptr;
    void* bytes = NULL;
    size_t size = h->core->savedataClone(h->core, &bytes);
    free(bytes);
    return size;
}
ARES_API int ares_save_battery(void* ptr, void* output, size_t size) {
    if (!ptr || !output) { set_error("Invalid battery output"); return 0; }
    AresCore* h = ptr;
    void* bytes = NULL;
    size_t actual = h->core->savedataClone(h->core, &bytes);
    if (!bytes || actual != size) {
        free(bytes); set_error("Battery size mismatch"); return 0;
    }
    memcpy(output, bytes, size);
    free(bytes);
    return 1;
}
ARES_API int ares_load_battery(void* ptr, const void* input, size_t size) {
    if (!ptr || !input || !size || size > 0x20000) {
        set_error("Invalid battery input"); return 0;
    }
    AresCore* h = ptr;
    return h->core->savedataRestore(h->core, input, size, false);
}
