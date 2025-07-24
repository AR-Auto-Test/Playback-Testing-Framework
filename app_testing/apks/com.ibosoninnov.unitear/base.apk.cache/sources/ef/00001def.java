package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class RuntimeAssetDef extends Table {
    public static void addRenderFlags(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addByte(0, (byte) i, 0);
    }

    public static void addRenderPriority(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addByte(1, (byte) i, 4);
    }

    public static int createRuntimeAssetDef(FlatBufferBuilder flatBufferBuilder, int i, int i2) {
        flatBufferBuilder.startObject(2);
        addRenderPriority(flatBufferBuilder, i2);
        addRenderFlags(flatBufferBuilder, i);
        return endRuntimeAssetDef(flatBufferBuilder);
    }

    public static int endRuntimeAssetDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static RuntimeAssetDef getRootAsRuntimeAssetDef(ByteBuffer byteBuffer) {
        return getRootAsRuntimeAssetDef(byteBuffer, new RuntimeAssetDef());
    }

    public static void startRuntimeAssetDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public RuntimeAssetDef __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
        int i2 = i - byteBuffer.getInt(i);
        this.vtable_start = i2;
        this.vtable_size = this.bb.getShort(i2);
    }

    public int renderFlags() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public int renderPriority() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos) & UnsignedBytes.MAX_VALUE;
        }
        return 4;
    }

    public static RuntimeAssetDef getRootAsRuntimeAssetDef(ByteBuffer byteBuffer, RuntimeAssetDef runtimeAssetDef) {
        return runtimeAssetDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}