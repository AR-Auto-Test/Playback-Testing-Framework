package com.google.ar.sceneform.lullmodel;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class Vec2 extends Struct {
    public static int createVec2(FlatBufferBuilder flatBufferBuilder, float f2, float f3) {
        flatBufferBuilder.prep(4, 8);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public Vec2 __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public float x() {
        return this.bb.getFloat(this.bb_pos + 0);
    }

    public float y() {
        return this.bb.getFloat(this.bb_pos + 4);
    }
}