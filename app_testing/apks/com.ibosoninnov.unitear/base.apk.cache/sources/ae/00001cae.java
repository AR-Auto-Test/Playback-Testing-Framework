package com.google.ar.sceneform.lullmodel;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class Vec3 extends Struct {
    public static int createVec3(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4) {
        flatBufferBuilder.prep(4, 12);
        flatBufferBuilder.putFloat(f4);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public Vec3 __assign(int i, ByteBuffer byteBuffer) {
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

    public float z() {
        return this.bb.getFloat(this.bb_pos + 8);
    }
}