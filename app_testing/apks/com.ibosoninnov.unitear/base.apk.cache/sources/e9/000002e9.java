package b.d.b.d1;

import b.d.b.d1.i0;
import java.util.Set;

/* compiled from: ReadableConfig.java */
/* loaded from: classes.dex */
public interface a1 extends i0 {
    @Override // b.d.b.d1.i0
    default <ValueT> ValueT a(i0.a<ValueT> aVar) {
        return (ValueT) k().a(aVar);
    }

    @Override // b.d.b.d1.i0
    default boolean b(i0.a<?> aVar) {
        return k().b(aVar);
    }

    @Override // b.d.b.d1.i0
    default void c(String str, i0.b bVar) {
        k().c(str, bVar);
    }

    @Override // b.d.b.d1.i0
    default <ValueT> ValueT d(i0.a<ValueT> aVar, i0.c cVar) {
        return (ValueT) k().d(aVar, cVar);
    }

    @Override // b.d.b.d1.i0
    default Set<i0.a<?>> e() {
        return k().e();
    }

    @Override // b.d.b.d1.i0
    default <ValueT> ValueT f(i0.a<ValueT> aVar, ValueT valuet) {
        return (ValueT) k().f(aVar, valuet);
    }

    @Override // b.d.b.d1.i0
    default i0.c g(i0.a<?> aVar) {
        return k().g(aVar);
    }

    @Override // b.d.b.d1.i0
    default Set<i0.c> h(i0.a<?> aVar) {
        return k().h(aVar);
    }

    i0 k();
}