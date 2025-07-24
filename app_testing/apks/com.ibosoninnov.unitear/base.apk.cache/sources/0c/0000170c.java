package com.google.android.material.canvas;

import android.graphics.Canvas;
import android.graphics.RectF;

/* loaded from: classes.dex */
public class CanvasCompat {
    private CanvasCompat() {
    }

    public static int saveLayerAlpha(Canvas canvas, RectF rectF, int i) {
        return canvas.saveLayerAlpha(rectF, i);
    }

    public static int saveLayerAlpha(Canvas canvas, float f2, float f3, float f4, float f5, int i) {
        return canvas.saveLayerAlpha(f2, f3, f4, f5, i);
    }
}