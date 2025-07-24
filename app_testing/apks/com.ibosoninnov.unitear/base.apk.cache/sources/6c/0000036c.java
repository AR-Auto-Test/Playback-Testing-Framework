package b.d.b.e1;

import b.d.b.d1.a1;
import b.d.b.d1.i0;
import b.d.b.d1.n;

/* compiled from: TargetConfig.java */
/* loaded from: classes.dex */
public interface e<T> extends a1 {
    public static final i0.a<String> n = new n("camerax.core.target.name", String.class, null);
    public static final i0.a<Class<?>> o = new n("camerax.core.target.class", Class.class, null);

    /* JADX DEBUG: Type inference failed for r0v0. Raw type applied. Possible types: b.d.b.d1.i0$a<java.lang.String>, b.d.b.d1.i0$a<ValueT> */
    default String p(String str) {
        return (String) f(n, str);
    }
}