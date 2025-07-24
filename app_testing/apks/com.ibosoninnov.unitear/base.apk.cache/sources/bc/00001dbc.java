package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class AnimListFb extends Table {
    public static boolean AnimListFbBufferHasIdentifier(ByteBuffer byteBuffer) {
        return Table.__has_identifier(byteBuffer, "ALIS");
    }

    public static void addAnimFiles(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addAnims(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static int createAnimFilesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createAnimListFb(FlatBufferBuilder flatBufferBuilder, int i, int i2) {
        flatBufferBuilder.startObject(2);
        addAnims(flatBufferBuilder, i2);
        addAnimFiles(flatBufferBuilder, i);
        return endAnimListFb(flatBufferBuilder);
    }

    public static int createAnimsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endAnimListFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static void finishAnimListFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finish(i, "ALIS");
    }

    public static void finishSizePrefixedAnimListFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finishSizePrefixed(i, "ALIS");
    }

    public static AnimListFb getRootAsAnimListFb(ByteBuffer byteBuffer) {
        return getRootAsAnimListFb(byteBuffer, new AnimListFb());
    }

    public static void startAnimFilesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startAnimListFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public static void startAnimsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public AnimListFb __assign(int i, ByteBuffer byteBuffer) {
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

    public String animFiles(int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string((i * 4) + __vector(__offset));
        }
        return null;
    }

    public int animFilesLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public AnimSource anims(int i) {
        return anims(new AnimSource(), i);
    }

    public int animsLength() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static AnimListFb getRootAsAnimListFb(ByteBuffer byteBuffer, AnimListFb animListFb) {
        return animListFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public AnimSource anims(AnimSource animSource, int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return animSource.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}