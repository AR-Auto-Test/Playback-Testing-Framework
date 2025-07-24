package b.d.b;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;

/* compiled from: CameraSelector.java */
/* loaded from: classes.dex */
public final class j0 {

    /* renamed from: a  reason: collision with root package name */
    public static final j0 f1629a;

    /* renamed from: b  reason: collision with root package name */
    public static final j0 f1630b;

    /* renamed from: c  reason: collision with root package name */
    public LinkedHashSet<h0> f1631c;

    static {
        LinkedHashSet linkedHashSet = new LinkedHashSet();
        linkedHashSet.add(new b.d.b.d1.q0(0));
        f1629a = new j0(linkedHashSet);
        LinkedHashSet linkedHashSet2 = new LinkedHashSet();
        linkedHashSet2.add(new b.d.b.d1.q0(1));
        f1630b = new j0(linkedHashSet2);
    }

    public j0(LinkedHashSet<h0> linkedHashSet) {
        this.f1631c = linkedHashSet;
    }

    public LinkedHashSet<b.d.b.d1.a0> a(LinkedHashSet<b.d.b.d1.a0> linkedHashSet) {
        ArrayList arrayList = new ArrayList();
        Iterator<b.d.b.d1.a0> it = linkedHashSet.iterator();
        while (it.hasNext()) {
            arrayList.add(it.next().b());
        }
        List<i0> b2 = b(arrayList);
        LinkedHashSet<b.d.b.d1.a0> linkedHashSet2 = new LinkedHashSet<>();
        Iterator<b.d.b.d1.a0> it2 = linkedHashSet.iterator();
        while (it2.hasNext()) {
            b.d.b.d1.a0 next = it2.next();
            if (b2.contains(next.b())) {
                linkedHashSet2.add(next);
            }
        }
        return linkedHashSet2;
    }

    public List<i0> b(List<i0> list) {
        ArrayList arrayList = new ArrayList(list);
        List<i0> arrayList2 = new ArrayList<>(list);
        Iterator<h0> it = this.f1631c.iterator();
        while (it.hasNext()) {
            arrayList2 = it.next().a(Collections.unmodifiableList(arrayList2));
            if (!arrayList2.isEmpty()) {
                if (arrayList.containsAll(arrayList2)) {
                    arrayList.retainAll(arrayList2);
                } else {
                    throw new IllegalArgumentException("The output isn't contained in the input.");
                }
            } else {
                throw new IllegalArgumentException("No available camera can be found.");
            }
        }
        return arrayList2;
    }

    public Integer c() {
        Iterator<h0> it = this.f1631c.iterator();
        Integer num = null;
        while (it.hasNext()) {
            h0 next = it.next();
            if (next instanceof b.d.b.d1.q0) {
                Integer valueOf = Integer.valueOf(((b.d.b.d1.q0) next).f1585a);
                if (num == null) {
                    num = valueOf;
                } else if (!num.equals(valueOf)) {
                    throw new IllegalStateException("Multiple conflicting lens facing requirements exist.");
                }
            }
        }
        return num;
    }
}