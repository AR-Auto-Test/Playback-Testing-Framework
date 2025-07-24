package b.d.b.d1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/* compiled from: MultiValueSet.java */
/* loaded from: classes.dex */
public abstract class s0<C> {

    /* renamed from: a  reason: collision with root package name */
    public Set<C> f1588a = new HashSet();

    /* JADX DEBUG: Method merged with bridge method */
    @Override // 
    /* renamed from: a */
    public abstract s0<C> clone();

    public List<C> b() {
        return Collections.unmodifiableList(new ArrayList(this.f1588a));
    }
}