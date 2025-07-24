package com.google.ar.sceneform.rendering;

import android.view.View;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class DpToMetersViewSizer implements ViewSizer {
    private static final float DEFAULT_SIZE_Z = 0.0f;
    private final int dpPerMeters;

    public DpToMetersViewSizer(int i) {
        if (i > 0) {
            this.dpPerMeters = i;
            return;
        }
        throw new IllegalArgumentException("dpPerMeters must be greater than zero.");
    }

    public int getDpPerMeters() {
        return this.dpPerMeters;
    }

    @Override // com.google.ar.sceneform.rendering.ViewSizer
    public Vector3 getSize(View view) {
        Preconditions.checkNotNull(view, "Parameter \"view\" was null.");
        float convertPxToDp = ViewRenderableHelpers.convertPxToDp(view.getWidth());
        float convertPxToDp2 = ViewRenderableHelpers.convertPxToDp(view.getHeight());
        int i = this.dpPerMeters;
        return new Vector3(convertPxToDp / i, convertPxToDp2 / i, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }
}