package com.google.android.material.shape;

import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public class TriangleEdgeTreatment extends EdgeTreatment {
    private final boolean inside;
    private final float size;

    public TriangleEdgeTreatment(float f2, boolean z) {
        this.size = f2;
        this.inside = z;
    }

    @Override // com.google.android.material.shape.EdgeTreatment
    public void getEdgePath(float f2, float f3, float f4, ShapePath shapePath) {
        shapePath.lineTo(f3 - (this.size * f4), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        shapePath.lineTo(f3, (this.inside ? this.size : -this.size) * f4);
        shapePath.lineTo((this.size * f4) + f3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        shapePath.lineTo(f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }
}