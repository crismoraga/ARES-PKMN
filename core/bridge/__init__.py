"""Typed native and synthetic GBA backends."""
from core.bridge.ffi_mgba import GBABridgeProtocol, LowLevelMGBABridge, NativeLibMGBABridge
from core.bridge.synthetic_gba_silicon import SyntheticGBASiliconBridge

__all__ = ["GBABridgeProtocol", "LowLevelMGBABridge", "NativeLibMGBABridge", "SyntheticGBASiliconBridge"]
