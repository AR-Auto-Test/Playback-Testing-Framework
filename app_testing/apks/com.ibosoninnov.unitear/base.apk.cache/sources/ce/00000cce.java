package com.google.android.filament;

import com.google.android.filament.proguard.UsedByNative;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

@UsedByNative("NioUtils.cpp")
/* loaded from: classes.dex */
public final class NioUtils {

    /* loaded from: classes.dex */
    public enum BufferType {
        BYTE,
        CHAR,
        SHORT,
        INT,
        LONG,
        FLOAT,
        DOUBLE
    }

    private NioUtils() {
    }

    @UsedByNative("NioUtils.cpp")
    public static Object getBaseArray(Buffer buffer) {
        if (buffer.hasArray()) {
            return buffer.array();
        }
        return null;
    }

    @UsedByNative("NioUtils.cpp")
    public static int getBaseArrayOffset(Buffer buffer, int i) {
        if (buffer.hasArray()) {
            return (buffer.position() + buffer.arrayOffset()) << i;
        }
        return 0;
    }

    @UsedByNative("NioUtils.cpp")
    public static long getBasePointer(Buffer buffer, long j, int i) {
        if (j != 0) {
            return (buffer.position() << i) + j;
        }
        return 0L;
    }

    @UsedByNative("NioUtils.cpp")
    public static int getBufferType(Buffer buffer) {
        if (buffer instanceof ByteBuffer) {
            return BufferType.BYTE.ordinal();
        }
        if (buffer instanceof CharBuffer) {
            return BufferType.CHAR.ordinal();
        }
        if (buffer instanceof ShortBuffer) {
            return BufferType.SHORT.ordinal();
        }
        if (buffer instanceof IntBuffer) {
            return BufferType.INT.ordinal();
        }
        if (buffer instanceof LongBuffer) {
            return BufferType.LONG.ordinal();
        }
        if (buffer instanceof FloatBuffer) {
            return BufferType.FLOAT.ordinal();
        }
        return BufferType.DOUBLE.ordinal();
    }
}