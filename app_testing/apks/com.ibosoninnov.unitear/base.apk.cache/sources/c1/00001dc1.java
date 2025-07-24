package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class AnimTableFb extends Table {
    public static boolean AnimTableFbBufferHasIdentifier(ByteBuffer byteBuffer) {
        return Table.__has_identifier(byteBuffer, "ATAB");
    }

    public static void addLists(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createAnimTableFb(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addLists(flatBufferBuilder, i);
        return endAnimTableFb(flatBufferBuilder);
    }

    public static int createListsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endAnimTableFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static void finishAnimTableFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finish(i, "ATAB");
    }

    public static void finishSizePrefixedAnimTableFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finishSizePrefixed(i, "ATAB");
    }

    public static AnimTableFb getRootAsAnimTableFb(ByteBuffer byteBuffer) {
        return getRootAsAnimTableFb(byteBuffer, new AnimTableFb());
    }

    public static void startAnimTableFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public static void startListsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public AnimTableFb __assign(int i, ByteBuffer byteBuffer) {
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

    public AnimListFb lists(int i) {
        return lists(new AnimListFb(), i);
    }

    public int listsLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static AnimTableFb getRootAsAnimTableFb(ByteBuffer byteBuffer, AnimTableFb animTableFb) {
        return animTableFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public AnimListFb lists(AnimListFb animListFb, int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return animListFb.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}