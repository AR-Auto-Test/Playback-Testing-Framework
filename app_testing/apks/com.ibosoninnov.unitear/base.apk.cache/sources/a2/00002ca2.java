package com.ibosoninnov.unitear;

import android.content.Context;
import android.util.AttributeSet;
import android.view.TextureView;
import android.view.View;

/* loaded from: classes2.dex */
public class AutoFitTextureView extends TextureView {

    /* renamed from: b  reason: collision with root package name */
    public int f5663b;

    /* renamed from: c  reason: collision with root package name */
    public int f5664c;

    /* renamed from: d  reason: collision with root package name */
    public float f5665d;

    public AutoFitTextureView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, 0);
        this.f5663b = 0;
        this.f5664c = 0;
        this.f5665d = 1.66f;
    }

    public void a(int i, int i2) {
        if (i >= 0 && i2 >= 0) {
            this.f5663b = i;
            this.f5664c = i2;
            requestLayout();
            return;
        }
        throw new IllegalArgumentException("Size cannot be negative.");
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        int i3;
        super.onMeasure(i, i2);
        int size = (int) (View.MeasureSpec.getSize(i) * this.f5665d);
        int size2 = View.MeasureSpec.getSize(i2);
        int i4 = this.f5663b;
        if (i4 != 0 && (i3 = this.f5664c) != 0) {
            if (size < (size2 * i4) / i3) {
                setMeasuredDimension(size, (i3 * size) / i4);
                return;
            } else {
                setMeasuredDimension((i4 * size2) / i3, size2);
                return;
            }
        }
        setMeasuredDimension(size, size2);
    }
}