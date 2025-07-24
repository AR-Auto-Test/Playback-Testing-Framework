package com.google.android.filament;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/* loaded from: classes.dex */
public class Colors {

    /* renamed from: com.google.android.filament.Colors$1  reason: invalid class name */
    /* loaded from: classes.dex */
    public static /* synthetic */ class AnonymousClass1 {
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$Colors$Conversion;
        public static final /* synthetic */ int[] $SwitchMap$com$google$android$filament$Colors$RgbaType;

        static {
            Conversion.values();
            int[] iArr = new int[2];
            $SwitchMap$com$google$android$filament$Colors$Conversion = iArr;
            try {
                iArr[Conversion.ACCURATE.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                $SwitchMap$com$google$android$filament$Colors$Conversion[Conversion.FAST.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            RgbaType.values();
            int[] iArr2 = new int[4];
            $SwitchMap$com$google$android$filament$Colors$RgbaType = iArr2;
            try {
                iArr2[RgbaType.SRGB.ordinal()] = 1;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                $SwitchMap$com$google$android$filament$Colors$RgbaType[RgbaType.LINEAR.ordinal()] = 2;
            } catch (NoSuchFieldError unused4) {
            }
            try {
                $SwitchMap$com$google$android$filament$Colors$RgbaType[RgbaType.PREMULTIPLIED_SRGB.ordinal()] = 3;
            } catch (NoSuchFieldError unused5) {
            }
            try {
                $SwitchMap$com$google$android$filament$Colors$RgbaType[RgbaType.PREMULTIPLIED_LINEAR.ordinal()] = 4;
            } catch (NoSuchFieldError unused6) {
            }
        }
    }

    /* loaded from: classes.dex */
    public enum Conversion {
        ACCURATE,
        FAST
    }

    @Target({ElementType.PARAMETER, ElementType.METHOD, ElementType.LOCAL_VARIABLE, ElementType.FIELD})
    @Retention(RetentionPolicy.SOURCE)
    /* loaded from: classes.dex */
    public @interface LinearColor {
    }

    /* loaded from: classes.dex */
    public enum RgbType {
        SRGB,
        LINEAR
    }

    /* loaded from: classes.dex */
    public enum RgbaType {
        SRGB,
        LINEAR,
        PREMULTIPLIED_SRGB,
        PREMULTIPLIED_LINEAR
    }

    private Colors() {
    }

    public static float[] cct(float f2) {
        float[] fArr = new float[3];
        nCct(f2, fArr);
        return fArr;
    }

    public static float[] illuminantD(float f2) {
        float[] fArr = new float[3];
        nIlluminantD(f2, fArr);
        return fArr;
    }

    private static native void nCct(float f2, float[] fArr);

    private static native void nIlluminantD(float f2, float[] fArr);

    public static float[] toLinear(RgbType rgbType, float f2, float f3, float f4) {
        return toLinear(rgbType, new float[]{f2, f3, f4});
    }

    public static float[] toLinear(RgbType rgbType, float[] fArr) {
        return rgbType == RgbType.LINEAR ? fArr : toLinear(Conversion.ACCURATE, fArr);
    }

    public static float[] toLinear(RgbaType rgbaType, float f2, float f3, float f4, float f5) {
        return toLinear(rgbaType, new float[]{f2, f3, f4, f5});
    }

    public static float[] toLinear(RgbaType rgbaType, float[] fArr) {
        int ordinal = rgbaType.ordinal();
        if (ordinal == 0) {
            toLinear(Conversion.ACCURATE, fArr);
        } else if (ordinal != 1) {
            return ordinal != 2 ? fArr : toLinear(Conversion.ACCURATE, fArr);
        }
        float f2 = fArr[3];
        fArr[0] = fArr[0] * f2;
        fArr[1] = fArr[1] * f2;
        fArr[2] = fArr[2] * f2;
        return fArr;
    }

    public static float[] toLinear(Conversion conversion, float[] fArr) {
        int ordinal = conversion.ordinal();
        int i = 0;
        if (ordinal == 0) {
            while (i < 3) {
                fArr[i] = fArr[i] <= 0.04045f ? fArr[i] / 12.92f : (float) Math.pow((fArr[i] + 0.055f) / 1.055f, 2.4000000953674316d);
                i++;
            }
        } else if (ordinal == 1) {
            while (i < 3) {
                fArr[i] = (float) Math.sqrt(fArr[i]);
                i++;
            }
        }
        return fArr;
    }
}