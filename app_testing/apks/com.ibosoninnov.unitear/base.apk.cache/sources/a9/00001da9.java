package com.google.ar.schemas.lull;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class Rect extends Struct {
    public static int createRect(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4, float f5) {
        flatBufferBuilder.prep(4, 16);
        flatBufferBuilder.putFloat(f5);
        flatBufferBuilder.putFloat(f4);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public Rect __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public float h() {
        return this.bb.getFloat(this.bb_pos + 12);
    }

    public float w() {
        return this.bb.getFloat(this.bb_pos + 8);
    }

    public float x() {
        return this.bb.getFloat(this.bb_pos + 0);
    }

    public float y() {
        return this.bb.getFloat(this.bb_pos + 4);
    }
}