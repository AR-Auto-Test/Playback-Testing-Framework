package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class AnimSourceEmbedded extends Table {
    public static void addEmbedded(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createAnimSourceEmbedded(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addEmbedded(flatBufferBuilder, i);
        return endAnimSourceEmbedded(flatBufferBuilder);
    }

    public static int endAnimSourceEmbedded(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static AnimSourceEmbedded getRootAsAnimSourceEmbedded(ByteBuffer byteBuffer) {
        return getRootAsAnimSourceEmbedded(byteBuffer, new AnimSourceEmbedded());
    }

    public static void startAnimSourceEmbedded(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public AnimSourceEmbedded __assign(int i, ByteBuffer byteBuffer) {
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

    public RigAnimFb embedded() {
        return embedded(new RigAnimFb());
    }

    public static AnimSourceEmbedded getRootAsAnimSourceEmbedded(ByteBuffer byteBuffer, AnimSourceEmbedded animSourceEmbedded) {
        return animSourceEmbedded.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public RigAnimFb embedded(RigAnimFb rigAnimFb) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return rigAnimFb.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}