package c.a.a.z.l;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import c.a.a.j;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: NullLayer.java */
/* loaded from: classes.dex */
public class f extends b {
    public f(j jVar, e eVar) {
        super(jVar, eVar);
    }

    @Override // c.a.a.z.l.b, c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        super.d(rectF, matrix, z);
        rectF.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    @Override // c.a.a.z.l.b
    public void k(Canvas canvas, Matrix matrix, int i) {
    }
}