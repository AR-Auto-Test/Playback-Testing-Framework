package com.google.ar.schemas.lull;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class Color extends Struct {
    public static int createColor(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4, float f5) {
        flatBufferBuilder.prep(4, 16);
        flatBufferBuilder.putFloat(f5);
        flatBufferBuilder.putFloat(f4);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public Color __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public float a() {
        return this.bb.getFloat(this.bb_pos + 12);
    }

    public float b() {
        return this.bb.getFloat(this.bb_pos + 8);
    }

    public float g() {
        return this.bb.getFloat(this.bb_pos + 4);
    }

    public float r() {
        return this.bb.getFloat(this.bb_pos + 0);
    }
}