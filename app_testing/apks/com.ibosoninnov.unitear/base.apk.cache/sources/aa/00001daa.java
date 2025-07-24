package com.google.ar.schemas.lull;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class Recti extends Struct {
    public static int createRecti(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, int i4) {
        flatBufferBuilder.prep(4, 16);
        flatBufferBuilder.putInt(i4);
        flatBufferBuilder.putInt(i3);
        flatBufferBuilder.putInt(i2);
        flatBufferBuilder.putInt(i);
        return flatBufferBuilder.offset();
    }

    public Recti __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public int h() {
        return this.bb.getInt(this.bb_pos + 12);
    }

    public int w() {
        return this.bb.getInt(this.bb_pos + 8);
    }

    public int x() {
        return this.bb.getInt(this.bb_pos + 0);
    }

    public int y() {
        return this.bb.getInt(this.bb_pos + 4);
    }
}