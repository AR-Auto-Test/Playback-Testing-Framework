package com.google.flatbuffers;

import c.b.a.a.a;
import com.google.common.base.Ascii;
import java.nio.ByteBuffer;

/* loaded from: classes2.dex */
public abstract class Utf8 {
    private static Utf8 DEFAULT;

    /* loaded from: classes2.dex */
    public static class DecodeUtil {
        public static void handleFourBytes(byte b2, byte b3, byte b4, byte b5, char[] cArr, int i) {
            if (!isNotTrailingByte(b3)) {
                if ((((b3 + 112) + (b2 << Ascii.FS)) >> 30) == 0 && !isNotTrailingByte(b4) && !isNotTrailingByte(b5)) {
                    int trailingByteValue = ((b2 & 7) << 18) | (trailingByteValue(b3) << 12) | (trailingByteValue(b4) << 6) | trailingByteValue(b5);
                    cArr[i] = highSurrogate(trailingByteValue);
                    cArr[i + 1] = lowSurrogate(trailingByteValue);
                    return;
                }
            }
            throw new IllegalArgumentException("Invalid UTF-8");
        }

        public static void handleOneByte(byte b2, char[] cArr, int i) {
            cArr[i] = (char) b2;
        }

        public static void handleThreeBytes(byte b2, byte b3, byte b4, char[] cArr, int i) {
            if (!isNotTrailingByte(b3) && ((b2 != -32 || b3 >= -96) && ((b2 != -19 || b3 < -96) && !isNotTrailingByte(b4)))) {
                cArr[i] = (char) (((b2 & 15) << 12) | (trailingByteValue(b3) << 6) | trailingByteValue(b4));
                return;
            }
            throw new IllegalArgumentException("Invalid UTF-8");
        }

        public static void handleTwoBytes(byte b2, byte b3, char[] cArr, int i) {
            if (b2 >= -62) {
                if (!isNotTrailingByte(b3)) {
                    cArr[i] = (char) (((b2 & Ascii.US) << 6) | trailingByteValue(b3));
                    return;
                }
                throw new IllegalArgumentException("Invalid UTF-8: Illegal trailing byte in 2 bytes utf");
            }
            throw new IllegalArgumentException("Invalid UTF-8: Illegal leading byte in 2 bytes utf");
        }

        private static char highSurrogate(int i) {
            return (char) ((i >>> 10) + 55232);
        }

        private static boolean isNotTrailingByte(byte b2) {
            return b2 > -65;
        }

        public static boolean isOneByte(byte b2) {
            return b2 >= 0;
        }

        public static boolean isThreeBytes(byte b2) {
            return b2 < -16;
        }

        public static boolean isTwoBytes(byte b2) {
            return b2 < -32;
        }

        private static char lowSurrogate(int i) {
            return (char) ((i & 1023) + 56320);
        }

        private static int trailingByteValue(byte b2) {
            return b2 & 63;
        }
    }

    /* loaded from: classes2.dex */
    public static class UnpairedSurrogateException extends IllegalArgumentException {
        public UnpairedSurrogateException(int i, int i2) {
            super(a.k("Unpaired surrogate at index ", i, " of ", i2));
        }
    }

    public static Utf8 getDefault() {
        if (DEFAULT == null) {
            DEFAULT = new Utf8Safe();
        }
        return DEFAULT;
    }

    public static void setDefault(Utf8 utf8) {
        DEFAULT = utf8;
    }

    public abstract String decodeUtf8(ByteBuffer byteBuffer, int i, int i2);

    public abstract void encodeUtf8(CharSequence charSequence, ByteBuffer byteBuffer);

    public abstract int encodedLength(CharSequence charSequence);
}