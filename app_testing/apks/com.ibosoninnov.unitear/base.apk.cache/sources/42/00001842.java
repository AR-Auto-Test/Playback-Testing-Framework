package com.google.android.material.shape;

import android.graphics.RectF;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;

/* loaded from: classes.dex */
public final class AdjustedCornerSize implements CornerSize {
    private final float adjustment;
    private final CornerSize other;

    public AdjustedCornerSize(float f2, CornerSize cornerSize) {
        while (cornerSize instanceof AdjustedCornerSize) {
            cornerSize = ((AdjustedCornerSize) cornerSize).other;
            f2 += ((AdjustedCornerSize) cornerSize).adjustment;
        }
        this.other = cornerSize;
        this.adjustment = f2;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof AdjustedCornerSize) {
            AdjustedCornerSize adjustedCornerSize = (AdjustedCornerSize) obj;
            return this.other.equals(adjustedCornerSize.other) && this.adjustment == adjustedCornerSize.adjustment;
        }
        return false;
    }

    @Override // com.google.android.material.shape.CornerSize
    public float getCornerSize(RectF rectF) {
        return Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, this.other.getCornerSize(rectF) + this.adjustment);
    }

    public int hashCode() {
        return Arrays.hashCode(new Object[]{this.other, Float.valueOf(this.adjustment)});
    }
}