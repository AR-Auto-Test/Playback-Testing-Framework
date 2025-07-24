package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModelPipelineCollidableDef extends Table {
    public static void addSource(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createModelPipelineCollidableDef(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addSource(flatBufferBuilder, i);
        return endModelPipelineCollidableDef(flatBufferBuilder);
    }

    public static int endModelPipelineCollidableDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModelPipelineCollidableDef getRootAsModelPipelineCollidableDef(ByteBuffer byteBuffer) {
        return getRootAsModelPipelineCollidableDef(byteBuffer, new ModelPipelineCollidableDef());
    }

    public static void startModelPipelineCollidableDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public ModelPipelineCollidableDef __assign(int i, ByteBuffer byteBuffer) {
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

    public String source() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer sourceAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer sourceInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public static ModelPipelineCollidableDef getRootAsModelPipelineCollidableDef(ByteBuffer byteBuffer, ModelPipelineCollidableDef modelPipelineCollidableDef) {
        return modelPipelineCollidableDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}