package b.y.a;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.drawable.Animatable;
import android.graphics.drawable.Drawable;
import android.view.animation.Interpolator;
import android.view.animation.LinearInterpolator;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Objects;

/* compiled from: CircularProgressDrawable.java */
/* loaded from: classes.dex */
public class d extends Drawable implements Animatable {

    /* renamed from: b  reason: collision with root package name */
    public static final Interpolator f2833b = new LinearInterpolator();

    /* renamed from: c  reason: collision with root package name */
    public static final Interpolator f2834c = new b.r.a.a.b();

    /* renamed from: d  reason: collision with root package name */
    public static final int[] f2835d = {-16777216};

    /* renamed from: e  reason: collision with root package name */
    public final a f2836e;

    /* renamed from: f  reason: collision with root package name */
    public float f2837f;

    /* renamed from: g  reason: collision with root package name */
    public Resources f2838g;

    /* renamed from: h  reason: collision with root package name */
    public Animator f2839h;
    public float i;
    public boolean j;

    /* compiled from: CircularProgressDrawable.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final RectF f2840a = new RectF();

        /* renamed from: b  reason: collision with root package name */
        public final Paint f2841b;

        /* renamed from: c  reason: collision with root package name */
        public final Paint f2842c;

        /* renamed from: d  reason: collision with root package name */
        public final Paint f2843d;

        /* renamed from: e  reason: collision with root package name */
        public float f2844e;

        /* renamed from: f  reason: collision with root package name */
        public float f2845f;

        /* renamed from: g  reason: collision with root package name */
        public float f2846g;

        /* renamed from: h  reason: collision with root package name */
        public float f2847h;
        public int[] i;
        public int j;
        public float k;
        public float l;
        public float m;
        public boolean n;
        public Path o;
        public float p;
        public float q;
        public int r;
        public int s;
        public int t;
        public int u;

        public a() {
            Paint paint = new Paint();
            this.f2841b = paint;
            Paint paint2 = new Paint();
            this.f2842c = paint2;
            Paint paint3 = new Paint();
            this.f2843d = paint3;
            this.f2844e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.f2845f = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.f2846g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.f2847h = 5.0f;
            this.p = 1.0f;
            this.t = 255;
            paint.setStrokeCap(Paint.Cap.SQUARE);
            paint.setAntiAlias(true);
            paint.setStyle(Paint.Style.STROKE);
            paint2.setStyle(Paint.Style.FILL);
            paint2.setAntiAlias(true);
            paint3.setColor(0);
        }

        public void a(int i) {
            this.j = i;
            this.u = this.i[i];
        }

        public void b(boolean z) {
            if (this.n != z) {
                this.n = z;
            }
        }
    }

    public d(Context context) {
        Objects.requireNonNull(context);
        this.f2838g = context.getResources();
        a aVar = new a();
        this.f2836e = aVar;
        aVar.i = f2835d;
        aVar.a(0);
        aVar.f2847h = 2.5f;
        aVar.f2841b.setStrokeWidth(2.5f);
        invalidateSelf();
        ValueAnimator ofFloat = ValueAnimator.ofFloat(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
        ofFloat.addUpdateListener(new b(this, aVar));
        ofFloat.setRepeatCount(-1);
        ofFloat.setRepeatMode(1);
        ofFloat.setInterpolator(f2833b);
        ofFloat.addListener(new c(this, aVar));
        this.f2839h = ofFloat;
    }

    public void a(float f2, a aVar, boolean z) {
        float interpolation;
        float f3;
        if (this.j) {
            d(f2, aVar);
            float floor = (float) (Math.floor(aVar.m / 0.8f) + 1.0d);
            float f4 = aVar.k;
            float f5 = aVar.l;
            aVar.f2844e = (((f5 - 0.01f) - f4) * f2) + f4;
            aVar.f2845f = f5;
            float f6 = aVar.m;
            aVar.f2846g = c.b.a.a.a.a(floor, f6, f2, f6);
        } else if (f2 != 1.0f || z) {
            float f7 = aVar.m;
            if (f2 < 0.5f) {
                interpolation = aVar.k;
                f3 = (f2834c.getInterpolation(f2 / 0.5f) * 0.79f) + 0.01f + interpolation;
            } else {
                float f8 = aVar.k + 0.79f;
                interpolation = f8 - (((1.0f - f2834c.getInterpolation((f2 - 0.5f) / 0.5f)) * 0.79f) + 0.01f);
                f3 = f8;
            }
            aVar.f2844e = interpolation;
            aVar.f2845f = f3;
            aVar.f2846g = (0.20999998f * f2) + f7;
            this.f2837f = (f2 + this.i) * 216.0f;
        }
    }

    public final void b(float f2, float f3, float f4, float f5) {
        a aVar = this.f2836e;
        float f6 = this.f2838g.getDisplayMetrics().density;
        float f7 = f3 * f6;
        aVar.f2847h = f7;
        aVar.f2841b.setStrokeWidth(f7);
        aVar.q = f2 * f6;
        aVar.a(0);
        aVar.r = (int) (f4 * f6);
        aVar.s = (int) (f5 * f6);
    }

    public void c(int i) {
        if (i == 0) {
            b(11.0f, 3.0f, 12.0f, 6.0f);
        } else {
            b(7.5f, 2.5f, 10.0f, 5.0f);
        }
        invalidateSelf();
    }

    public void d(float f2, a aVar) {
        if (f2 > 0.75f) {
            float f3 = (f2 - 0.75f) / 0.25f;
            int[] iArr = aVar.i;
            int i = aVar.j;
            int i2 = iArr[i];
            int i3 = iArr[(i + 1) % iArr.length];
            int i4 = (i2 >> 24) & 255;
            int i5 = (i2 >> 16) & 255;
            int i6 = (i2 >> 8) & 255;
            int i7 = i2 & 255;
            aVar.u = ((i4 + ((int) ((((i3 >> 24) & 255) - i4) * f3))) << 24) | ((i5 + ((int) ((((i3 >> 16) & 255) - i5) * f3))) << 16) | ((i6 + ((int) ((((i3 >> 8) & 255) - i6) * f3))) << 8) | (i7 + ((int) (f3 * ((i3 & 255) - i7))));
            return;
        }
        aVar.u = aVar.i[aVar.j];
    }

    @Override // android.graphics.drawable.Drawable
    public void draw(Canvas canvas) {
        Rect bounds = getBounds();
        canvas.save();
        canvas.rotate(this.f2837f, bounds.exactCenterX(), bounds.exactCenterY());
        a aVar = this.f2836e;
        RectF rectF = aVar.f2840a;
        float f2 = aVar.q;
        float f3 = (aVar.f2847h / 2.0f) + f2;
        if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            f3 = (Math.min(bounds.width(), bounds.height()) / 2.0f) - Math.max((aVar.r * aVar.p) / 2.0f, aVar.f2847h / 2.0f);
        }
        rectF.set(bounds.centerX() - f3, bounds.centerY() - f3, bounds.centerX() + f3, bounds.centerY() + f3);
        float f4 = aVar.f2844e;
        float f5 = aVar.f2846g;
        float f6 = (f4 + f5) * 360.0f;
        float f7 = ((aVar.f2845f + f5) * 360.0f) - f6;
        aVar.f2841b.setColor(aVar.u);
        aVar.f2841b.setAlpha(aVar.t);
        float f8 = aVar.f2847h / 2.0f;
        rectF.inset(f8, f8);
        canvas.drawCircle(rectF.centerX(), rectF.centerY(), rectF.width() / 2.0f, aVar.f2843d);
        float f9 = -f8;
        rectF.inset(f9, f9);
        canvas.drawArc(rectF, f6, f7, false, aVar.f2841b);
        if (aVar.n) {
            Path path = aVar.o;
            if (path == null) {
                Path path2 = new Path();
                aVar.o = path2;
                path2.setFillType(Path.FillType.EVEN_ODD);
            } else {
                path.reset();
            }
            aVar.o.moveTo(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            aVar.o.lineTo(aVar.r * aVar.p, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Path path3 = aVar.o;
            float f10 = aVar.p;
            path3.lineTo((aVar.r * f10) / 2.0f, aVar.s * f10);
            aVar.o.offset((rectF.centerX() + (Math.min(rectF.width(), rectF.height()) / 2.0f)) - ((aVar.r * aVar.p) / 2.0f), (aVar.f2847h / 2.0f) + rectF.centerY());
            aVar.o.close();
            aVar.f2842c.setColor(aVar.u);
            aVar.f2842c.setAlpha(aVar.t);
            canvas.save();
            canvas.rotate(f6 + f7, rectF.centerX(), rectF.centerY());
            canvas.drawPath(aVar.o, aVar.f2842c);
            canvas.restore();
        }
        canvas.restore();
    }

    @Override // android.graphics.drawable.Drawable
    public int getAlpha() {
        return this.f2836e.t;
    }

    @Override // android.graphics.drawable.Drawable
    public int getOpacity() {
        return -3;
    }

    @Override // android.graphics.drawable.Animatable
    public boolean isRunning() {
        return this.f2839h.isRunning();
    }

    @Override // android.graphics.drawable.Drawable
    public void setAlpha(int i) {
        this.f2836e.t = i;
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Drawable
    public void setColorFilter(ColorFilter colorFilter) {
        this.f2836e.f2841b.setColorFilter(colorFilter);
        invalidateSelf();
    }

    @Override // android.graphics.drawable.Animatable
    public void start() {
        this.f2839h.cancel();
        a aVar = this.f2836e;
        float f2 = aVar.f2844e;
        aVar.k = f2;
        float f3 = aVar.f2845f;
        aVar.l = f3;
        aVar.m = aVar.f2846g;
        if (f3 != f2) {
            this.j = true;
            this.f2839h.setDuration(666L);
            this.f2839h.start();
            return;
        }
        aVar.a(0);
        a aVar2 = this.f2836e;
        aVar2.k = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar2.l = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar2.m = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar2.f2844e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar2.f2845f = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar2.f2846g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.f2839h.setDuration(1332L);
        this.f2839h.start();
    }

    @Override // android.graphics.drawable.Animatable
    public void stop() {
        this.f2839h.cancel();
        this.f2837f = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.f2836e.b(false);
        this.f2836e.a(0);
        a aVar = this.f2836e;
        aVar.k = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar.l = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar.m = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar.f2844e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar.f2845f = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar.f2846g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        invalidateSelf();
    }
}