package com.google.android.gms.internal.vision;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import java.nio.ByteBuffer;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzw {
    public static Bitmap zza(Bitmap bitmap, zzs zzsVar) {
        int i;
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        if (zzsVar.zze != 0) {
            Matrix matrix = new Matrix();
            int i2 = zzsVar.zze;
            if (i2 == 0) {
                i = 0;
            } else if (i2 == 1) {
                i = 90;
            } else if (i2 == 2) {
                i = BaseTransientBottomBar.ANIMATION_FADE_DURATION;
            } else if (i2 != 3) {
                throw new IllegalArgumentException("Unsupported rotation degree.");
            } else {
                i = 270;
            }
            matrix.postRotate(i);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false);
        }
        int i3 = zzsVar.zze;
        if (i3 == 1 || i3 == 3) {
            zzsVar.zza = height;
            zzsVar.zzb = width;
        }
        return bitmap;
    }

    public static ByteBuffer zza(Bitmap bitmap, boolean z) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int i = width * height;
        ByteBuffer allocateDirect = ByteBuffer.allocateDirect(((((height + 1) / 2) * ((width + 1) / 2)) << 1) + i);
        int i2 = i;
        for (int i3 = 0; i3 < i; i3++) {
            int i4 = i3 % width;
            int i5 = i3 / width;
            int pixel = bitmap.getPixel(i4, i5);
            float red = Color.red(pixel);
            float green = Color.green(pixel);
            float blue = Color.blue(pixel);
            allocateDirect.put(i3, (byte) ((0.114f * blue) + (0.587f * green) + (0.299f * red)));
            if (i5 % 2 == 0 && i4 % 2 == 0) {
                float f2 = (-0.331f) * green;
                float f3 = blue * 0.5f;
                float f4 = blue * (-0.081f);
                int i6 = i2 + 1;
                allocateDirect.put(i2, (byte) (f3 + f2 + ((-0.169f) * red) + 128.0f));
                i2 = i6 + 1;
                allocateDirect.put(i6, (byte) (f4 + (green * (-0.419f)) + (red * 0.5f) + 128.0f));
            }
        }
        return allocateDirect;
    }
}