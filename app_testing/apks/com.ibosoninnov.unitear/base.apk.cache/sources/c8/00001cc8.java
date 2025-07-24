package com.google.ar.sceneform.rendering;

import android.view.View;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class FixedWidthViewSizer implements ViewSizer {
    private static final float DEFAULT_SIZE_Z = 0.0f;
    private final float widthMeters;

    public FixedWidthViewSizer(float f2) {
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            this.widthMeters = f2;
            return;
        }
        throw new IllegalArgumentException("widthMeters must be greater than zero.");
    }

    @Override // com.google.ar.sceneform.rendering.ViewSizer
    public Vector3 getSize(View view) {
        Preconditions.checkNotNull(view, "Parameter \"view\" was null.");
        float aspectRatio = ViewRenderableHelpers.getAspectRatio(view);
        if (aspectRatio == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return Vector3.zero();
        }
        float f2 = this.widthMeters;
        return new Vector3(f2, f2 / aspectRatio, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public float getWidth() {
        return this.widthMeters;
    }
}