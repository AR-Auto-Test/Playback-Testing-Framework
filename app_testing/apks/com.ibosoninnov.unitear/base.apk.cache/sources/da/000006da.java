package c.a.a.z.l;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import c.a.a.j;
import c.a.a.o;
import c.a.a.x.c.p;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: SolidLayer.java */
/* loaded from: classes.dex */
public class h extends b {
    public final Path A;
    public final e B;
    public c.a.a.x.c.a<ColorFilter, ColorFilter> C;
    public final RectF x;
    public final Paint y;
    public final float[] z;

    public h(j jVar, e eVar) {
        super(jVar, eVar);
        this.x = new RectF();
        c.a.a.x.a aVar = new c.a.a.x.a();
        this.y = aVar;
        this.z = new float[8];
        this.A = new Path();
        this.B = eVar;
        aVar.setAlpha(0);
        aVar.setStyle(Paint.Style.FILL);
        aVar.setColor(eVar.l);
    }

    @Override // c.a.a.z.l.b, c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        super.d(rectF, matrix, z);
        RectF rectF2 = this.x;
        e eVar = this.B;
        rectF2.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, eVar.j, eVar.k);
        this.m.mapRect(this.x);
        rectF.set(this.x);
    }

    @Override // c.a.a.z.l.b, c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        this.v.c(t, cVar);
        if (t == o.C) {
            if (cVar == null) {
                this.C = null;
            } else {
                this.C = new p(cVar, null);
            }
        }
    }

    @Override // c.a.a.z.l.b
    public void k(Canvas canvas, Matrix matrix, int i) {
        int alpha = Color.alpha(this.B.l);
        if (alpha == 0) {
            return;
        }
        c.a.a.x.c.a<Integer, Integer> aVar = this.v.j;
        int intValue = (int) ((((alpha / 255.0f) * (aVar == null ? 100 : aVar.e().intValue())) / 100.0f) * (i / 255.0f) * 255.0f);
        this.y.setAlpha(intValue);
        c.a.a.x.c.a<ColorFilter, ColorFilter> aVar2 = this.C;
        if (aVar2 != null) {
            this.y.setColorFilter(aVar2.e());
        }
        if (intValue > 0) {
            float[] fArr = this.z;
            fArr[0] = 0.0f;
            fArr[1] = 0.0f;
            e eVar = this.B;
            int i2 = eVar.j;
            fArr[2] = i2;
            fArr[3] = 0.0f;
            fArr[4] = i2;
            int i3 = eVar.k;
            fArr[5] = i3;
            fArr[6] = 0.0f;
            fArr[7] = i3;
            matrix.mapPoints(fArr);
            this.A.reset();
            Path path = this.A;
            float[] fArr2 = this.z;
            path.moveTo(fArr2[0], fArr2[1]);
            Path path2 = this.A;
            float[] fArr3 = this.z;
            path2.lineTo(fArr3[2], fArr3[3]);
            Path path3 = this.A;
            float[] fArr4 = this.z;
            path3.lineTo(fArr4[4], fArr4[5]);
            Path path4 = this.A;
            float[] fArr5 = this.z;
            path4.lineTo(fArr5[6], fArr5[7]);
            Path path5 = this.A;
            float[] fArr6 = this.z;
            path5.lineTo(fArr6[0], fArr6[1]);
            this.A.close();
            canvas.drawPath(this.A, this.y);
        }
    }
}