package com.google.ar.sceneform.rendering;

import android.content.res.Resources;
import android.view.View;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public class ViewRenderableHelpers {
    public static float convertPxToDp(int i) {
        return i / (Resources.getSystem().getDisplayMetrics().xdpi / 160.0f);
    }

    public static float getAspectRatio(View view) {
        float width = view.getWidth();
        float height = view.getHeight();
        return (width == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || height == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : width / height;
    }
}