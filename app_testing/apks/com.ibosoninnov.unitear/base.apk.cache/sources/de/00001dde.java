package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class DoubleVec4Init extends Table {
    public static void addW(FlatBufferBuilder flatBufferBuilder, double d2) {
        flatBufferBuilder.addDouble(3, d2, ShadowDrawableWrapper.COS_45);
    }

    public static void addX(FlatBufferBuilder flatBufferBuilder, double d2) {
        flatBufferBuilder.addDouble(0, d2, ShadowDrawableWrapper.COS_45);
    }

    public static void addY(FlatBufferBuilder flatBufferBuilder, double d2) {
        flatBufferBuilder.addDouble(1, d2, ShadowDrawableWrapper.COS_45);
    }

    public static void addZ(FlatBufferBuilder flatBufferBuilder, double d2) {
        flatBufferBuilder.addDouble(2, d2, ShadowDrawableWrapper.COS_45);
    }

    public static int createDoubleVec4Init(FlatBufferBuilder flatBufferBuilder, double d2, double d3, double d4, double d5) {
        flatBufferBuilder.startObject(4);
        addW(flatBufferBuilder, d5);
        addZ(flatBufferBuilder, d4);
        addY(flatBufferBuilder, d3);
        addX(flatBufferBuilder, d2);
        return endDoubleVec4Init(flatBufferBuilder);
    }

    public static int endDoubleVec4Init(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static DoubleVec4Init getRootAsDoubleVec4Init(ByteBuffer byteBuffer) {
        return getRootAsDoubleVec4Init(byteBuffer, new DoubleVec4Init());
    }

    public static void startDoubleVec4Init(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(4);
    }

    public DoubleVec4Init __assign(int i, ByteBuffer byteBuffer) {
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

    public double w() {
        int __offset = __offset(10);
        return __offset != 0 ? this.bb.getDouble(__offset + this.bb_pos) : ShadowDrawableWrapper.COS_45;
    }

    public double x() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getDouble(__offset + this.bb_pos) : ShadowDrawableWrapper.COS_45;
    }

    public double y() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getDouble(__offset + this.bb_pos) : ShadowDrawableWrapper.COS_45;
    }

    public double z() {
        int __offset = __offset(8);
        return __offset != 0 ? this.bb.getDouble(__offset + this.bb_pos) : ShadowDrawableWrapper.COS_45;
    }

    public static DoubleVec4Init getRootAsDoubleVec4Init(ByteBuffer byteBuffer, DoubleVec4Init doubleVec4Init) {
        return doubleVec4Init.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}