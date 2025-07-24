package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class TwitchParameters extends Table {
    public static void addSettled(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createTwitchParameters(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addSettled(flatBufferBuilder, i);
        return endTwitchParameters(flatBufferBuilder);
    }

    public static int endTwitchParameters(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static TwitchParameters getRootAsTwitchParameters(ByteBuffer byteBuffer) {
        return getRootAsTwitchParameters(byteBuffer, new TwitchParameters());
    }

    public static void startTwitchParameters(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public TwitchParameters __assign(int i, ByteBuffer byteBuffer) {
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

    public Settled1fParameters settled() {
        return settled(new Settled1fParameters());
    }

    public static TwitchParameters getRootAsTwitchParameters(ByteBuffer byteBuffer, TwitchParameters twitchParameters) {
        return twitchParameters.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Settled1fParameters settled(Settled1fParameters settled1fParameters) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return settled1fParameters.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}