package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.widget.SeekBar;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: AppCompatSeekBarHelper.java */
/* loaded from: classes.dex */
public class u extends q {

    /* renamed from: d  reason: collision with root package name */
    public final SeekBar f928d;

    /* renamed from: e  reason: collision with root package name */
    public Drawable f929e;

    /* renamed from: f  reason: collision with root package name */
    public ColorStateList f930f;

    /* renamed from: g  reason: collision with root package name */
    public PorterDuff.Mode f931g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f932h;
    public boolean i;

    public u(SeekBar seekBar) {
        super(seekBar);
        this.f930f = null;
        this.f931g = null;
        this.f932h = false;
        this.i = false;
        this.f928d = seekBar;
    }

    @Override // b.b.h.q
    public void a(AttributeSet attributeSet, int i) {
        super.a(attributeSet, i);
        Context context = this.f928d.getContext();
        int[] iArr = b.b.b.f547g;
        y0 r = y0.r(context, attributeSet, iArr, i, 0);
        SeekBar seekBar = this.f928d;
        b.j.j.q.m(seekBar, seekBar.getContext(), iArr, attributeSet, r.f972b, i, 0);
        Drawable h2 = r.h(0);
        if (h2 != null) {
            this.f928d.setThumb(h2);
        }
        Drawable g2 = r.g(1);
        Drawable drawable = this.f929e;
        if (drawable != null) {
            drawable.setCallback(null);
        }
        this.f929e = g2;
        if (g2 != null) {
            g2.setCallback(this.f928d);
            SeekBar seekBar2 = this.f928d;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            g2.setLayoutDirection(seekBar2.getLayoutDirection());
            if (g2.isStateful()) {
                g2.setState(this.f928d.getDrawableState());
            }
            c();
        }
        this.f928d.invalidate();
        if (r.p(3)) {
            this.f931g = e0.c(r.j(3, -1), this.f931g);
            this.i = true;
        }
        if (r.p(2)) {
            this.f930f = r.c(2);
            this.f932h = true;
        }
        r.f972b.recycle();
        c();
    }

    public final void c() {
        Drawable drawable = this.f929e;
        if (drawable != null) {
            if (this.f932h || this.i) {
                Drawable mutate = drawable.mutate();
                this.f929e = mutate;
                if (this.f932h) {
                    mutate.setTintList(this.f930f);
                }
                if (this.i) {
                    this.f929e.setTintMode(this.f931g);
                }
                if (this.f929e.isStateful()) {
                    this.f929e.setState(this.f928d.getDrawableState());
                }
            }
        }
    }

    public void d(Canvas canvas) {
        if (this.f929e != null) {
            int max = this.f928d.getMax();
            if (max > 1) {
                int intrinsicWidth = this.f929e.getIntrinsicWidth();
                int intrinsicHeight = this.f929e.getIntrinsicHeight();
                int i = intrinsicWidth >= 0 ? intrinsicWidth / 2 : 1;
                int i2 = intrinsicHeight >= 0 ? intrinsicHeight / 2 : 1;
                this.f929e.setBounds(-i, -i2, i, i2);
                float width = ((this.f928d.getWidth() - this.f928d.getPaddingLeft()) - this.f928d.getPaddingRight()) / max;
                int save = canvas.save();
                canvas.translate(this.f928d.getPaddingLeft(), this.f928d.getHeight() / 2);
                for (int i3 = 0; i3 <= max; i3++) {
                    this.f929e.draw(canvas);
                    canvas.translate(width, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                }
                canvas.restoreToCount(save);
            }
        }
    }
}