package com.google.ar.sceneform.rendering;

import com.google.android.filament.Colors;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public class Color {
    private static final float INT_COLOR_SCALE = 0.003921569f;

    /* renamed from: a  reason: collision with root package name */
    public float f5626a;

    /* renamed from: b  reason: collision with root package name */
    public float f5627b;

    /* renamed from: g  reason: collision with root package name */
    public float f5628g;
    public float r;

    public Color() {
        setWhite();
    }

    private static float inverseTonemap(float f2) {
        return ((-0.155f) * f2) / (f2 - 1.019f);
    }

    private void setWhite() {
        set(1.0f, 1.0f, 1.0f);
    }

    public Color inverseTonemap() {
        Color color = new Color(this.r, this.f5628g, this.f5627b, this.f5626a);
        color.r = inverseTonemap(this.r);
        color.f5628g = inverseTonemap(this.f5628g);
        color.f5627b = inverseTonemap(this.f5627b);
        return color;
    }

    public void set(Color color) {
        set(color.r, color.f5628g, color.f5627b, color.f5626a);
    }

    public void set(float f2, float f3, float f4) {
        set(f2, f3, f4, 1.0f);
    }

    public Color(Color color) {
        set(color);
    }

    public void set(float f2, float f3, float f4, float f5) {
        this.r = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Math.min(1.0f, f2));
        this.f5628g = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Math.min(1.0f, f3));
        this.f5627b = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Math.min(1.0f, f4));
        this.f5626a = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Math.min(1.0f, f5));
    }

    public Color(float f2, float f3, float f4) {
        set(f2, f3, f4);
    }

    public Color(float f2, float f3, float f4, float f5) {
        set(f2, f3, f4, f5);
    }

    public void set(int i) {
        int red = android.graphics.Color.red(i);
        int green = android.graphics.Color.green(i);
        int blue = android.graphics.Color.blue(i);
        int alpha = android.graphics.Color.alpha(i);
        float[] linear = Colors.toLinear(Colors.RgbType.SRGB, red * INT_COLOR_SCALE, green * INT_COLOR_SCALE, blue * INT_COLOR_SCALE);
        this.r = linear[0];
        this.f5628g = linear[1];
        this.f5627b = linear[2];
        this.f5626a = alpha * INT_COLOR_SCALE;
    }

    public Color(int i) {
        set(i);
    }
}