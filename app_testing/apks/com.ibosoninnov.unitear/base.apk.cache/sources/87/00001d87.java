package com.google.ar.schemas.lull;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Struct;
import java.nio.ByteBuffer;

/* loaded from: classes.dex */
public final class ArcDef extends Struct {
    public static int createArcDef(FlatBufferBuilder flatBufferBuilder, float f2, float f3, float f4, float f5, int i) {
        flatBufferBuilder.prep(4, 20);
        flatBufferBuilder.putInt(i);
        flatBufferBuilder.putFloat(f5);
        flatBufferBuilder.putFloat(f4);
        flatBufferBuilder.putFloat(f3);
        flatBufferBuilder.putFloat(f2);
        return flatBufferBuilder.offset();
    }

    public ArcDef __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
    }

    public float angleSize() {
        return this.bb.getFloat(this.bb_pos + 4);
    }

    public float innerRadius() {
        return this.bb.getFloat(this.bb_pos + 8);
    }

    public int numSamples() {
        return this.bb.getInt(this.bb_pos + 16);
    }

    public float outerRadius() {
        return this.bb.getFloat(this.bb_pos + 12);
    }

    public float startAngle() {
        return this.bb.getFloat(this.bb_pos + 0);
    }
}