package b.d.b.d1;

import com.google.auto.value.AutoValue;
import java.util.Set;

/* compiled from: Config.java */
/* loaded from: classes.dex */
public interface i0 {

    /* compiled from: Config.java */
    @AutoValue
    /* loaded from: classes.dex */
    public static abstract class a<T> {
        public abstract String a();

        public abstract Object b();

        public abstract Class<T> c();
    }

    /* compiled from: Config.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    /* compiled from: Config.java */
    /* loaded from: classes.dex */
    public enum c {
        ALWAYS_OVERRIDE,
        REQUIRED,
        OPTIONAL
    }

    <ValueT> ValueT a(a<ValueT> aVar);

    boolean b(a<?> aVar);

    void c(String str, b bVar);

    <ValueT> ValueT d(a<ValueT> aVar, c cVar);

    Set<a<?>> e();

    <ValueT> ValueT f(a<ValueT> aVar, ValueT valuet);

    c g(a<?> aVar);

    Set<c> h(a<?> aVar);
}