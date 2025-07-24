package c.a.a.x.c;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Collections;

/* compiled from: ValueCallbackKeyframeAnimation.java */
/* loaded from: classes.dex */
public class p<K, A> extends a<K, A> {
    public final A i;

    public p(c.a.a.d0.c<A> cVar, A a2) {
        super(Collections.emptyList());
        this.f3227e = cVar;
        this.i = a2;
    }

    @Override // c.a.a.x.c.a
    public float b() {
        return 1.0f;
    }

    @Override // c.a.a.x.c.a
    public A e() {
        c.a.a.d0.c<A> cVar = this.f3227e;
        A a2 = this.i;
        float f2 = this.f3226d;
        return cVar.a(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, a2, a2, f2, f2, f2);
    }

    @Override // c.a.a.x.c.a
    public A f(c.a.a.d0.a<K> aVar, float f2) {
        return e();
    }

    @Override // c.a.a.x.c.a
    public void g() {
        if (this.f3227e != null) {
            super.g();
        }
    }

    @Override // c.a.a.x.c.a
    public void h(float f2) {
        this.f3226d = f2;
    }
}