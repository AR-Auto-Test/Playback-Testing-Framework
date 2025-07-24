package b.w.b;

import androidx.recyclerview.widget.RecyclerView;
import b.w.b.p;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* compiled from: AdapterHelper.java */
/* loaded from: classes.dex */
public class a implements p.a {

    /* renamed from: d  reason: collision with root package name */
    public final InterfaceC0053a f2705d;

    /* renamed from: a  reason: collision with root package name */
    public b.j.i.d<b> f2702a = new b.j.i.e(30);

    /* renamed from: b  reason: collision with root package name */
    public final ArrayList<b> f2703b = new ArrayList<>();

    /* renamed from: c  reason: collision with root package name */
    public final ArrayList<b> f2704c = new ArrayList<>();

    /* renamed from: f  reason: collision with root package name */
    public int f2707f = 0;

    /* renamed from: e  reason: collision with root package name */
    public final p f2706e = new p(this);

    /* compiled from: AdapterHelper.java */
    /* renamed from: b.w.b.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0053a {
    }

    /* compiled from: AdapterHelper.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public int f2708a;

        /* renamed from: b  reason: collision with root package name */
        public int f2709b;

        /* renamed from: c  reason: collision with root package name */
        public Object f2710c;

        /* renamed from: d  reason: collision with root package name */
        public int f2711d;

        public b(int i, int i2, int i3, Object obj) {
            this.f2708a = i;
            this.f2709b = i2;
            this.f2711d = i3;
            this.f2710c = obj;
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null || b.class != obj.getClass()) {
                return false;
            }
            b bVar = (b) obj;
            int i = this.f2708a;
            if (i != bVar.f2708a) {
                return false;
            }
            if (i == 8 && Math.abs(this.f2711d - this.f2709b) == 1 && this.f2711d == bVar.f2709b && this.f2709b == bVar.f2711d) {
                return true;
            }
            if (this.f2711d == bVar.f2711d && this.f2709b == bVar.f2709b) {
                Object obj2 = this.f2710c;
                if (obj2 != null) {
                    if (!obj2.equals(bVar.f2710c)) {
                        return false;
                    }
                } else if (bVar.f2710c != null) {
                    return false;
                }
                return true;
            }
            return false;
        }

        public int hashCode() {
            return (((this.f2708a * 31) + this.f2709b) * 31) + this.f2711d;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(Integer.toHexString(System.identityHashCode(this)));
            sb.append("[");
            int i = this.f2708a;
            sb.append(i != 1 ? i != 2 ? i != 4 ? i != 8 ? "??" : "mv" : "up" : "rm" : "add");
            sb.append(",s:");
            sb.append(this.f2709b);
            sb.append("c:");
            sb.append(this.f2711d);
            sb.append(",p:");
            return c.b.a.a.a.u(sb, this.f2710c, "]");
        }
    }

    public a(InterfaceC0053a interfaceC0053a) {
        this.f2705d = interfaceC0053a;
    }

    public final boolean a(int i) {
        int size = this.f2704c.size();
        for (int i2 = 0; i2 < size; i2++) {
            b bVar = this.f2704c.get(i2);
            int i3 = bVar.f2708a;
            if (i3 == 8) {
                if (f(bVar.f2711d, i2 + 1) == i) {
                    return true;
                }
            } else if (i3 == 1) {
                int i4 = bVar.f2709b;
                int i5 = bVar.f2711d + i4;
                while (i4 < i5) {
                    if (f(i4, i2 + 1) == i) {
                        return true;
                    }
                    i4++;
                }
                continue;
            } else {
                continue;
            }
        }
        return false;
    }

    public void b() {
        int size = this.f2704c.size();
        for (int i = 0; i < size; i++) {
            ((RecyclerView.f) this.f2705d).a(this.f2704c.get(i));
        }
        l(this.f2704c);
        this.f2707f = 0;
    }

    public void c() {
        b();
        int size = this.f2703b.size();
        for (int i = 0; i < size; i++) {
            b bVar = this.f2703b.get(i);
            int i2 = bVar.f2708a;
            if (i2 == 1) {
                ((RecyclerView.f) this.f2705d).a(bVar);
                RecyclerView.f fVar = (RecyclerView.f) this.f2705d;
                RecyclerView.this.offsetPositionRecordsForInsert(bVar.f2709b, bVar.f2711d);
                RecyclerView.this.mItemsAddedOrRemoved = true;
            } else if (i2 == 2) {
                ((RecyclerView.f) this.f2705d).a(bVar);
                InterfaceC0053a interfaceC0053a = this.f2705d;
                int i3 = bVar.f2709b;
                int i4 = bVar.f2711d;
                RecyclerView.f fVar2 = (RecyclerView.f) interfaceC0053a;
                RecyclerView.this.offsetPositionRecordsForRemove(i3, i4, true);
                RecyclerView recyclerView = RecyclerView.this;
                recyclerView.mItemsAddedOrRemoved = true;
                recyclerView.mState.f392c += i4;
            } else if (i2 == 4) {
                ((RecyclerView.f) this.f2705d).a(bVar);
                ((RecyclerView.f) this.f2705d).c(bVar.f2709b, bVar.f2711d, bVar.f2710c);
            } else if (i2 == 8) {
                ((RecyclerView.f) this.f2705d).a(bVar);
                RecyclerView.f fVar3 = (RecyclerView.f) this.f2705d;
                RecyclerView.this.offsetPositionRecordsForMove(bVar.f2709b, bVar.f2711d);
                RecyclerView.this.mItemsAddedOrRemoved = true;
            }
        }
        l(this.f2703b);
        this.f2707f = 0;
    }

    public final void d(b bVar) {
        int i;
        int i2 = bVar.f2708a;
        if (i2 != 1 && i2 != 8) {
            int m = m(bVar.f2709b, i2);
            int i3 = bVar.f2709b;
            int i4 = bVar.f2708a;
            if (i4 == 2) {
                i = 0;
            } else if (i4 != 4) {
                throw new IllegalArgumentException("op should be remove or update." + bVar);
            } else {
                i = 1;
            }
            int i5 = 1;
            for (int i6 = 1; i6 < bVar.f2711d; i6++) {
                int m2 = m((i * i6) + bVar.f2709b, bVar.f2708a);
                int i7 = bVar.f2708a;
                if (i7 == 2 ? m2 == m : i7 == 4 && m2 == m + 1) {
                    i5++;
                } else {
                    b h2 = h(i7, m, i5, bVar.f2710c);
                    e(h2, i3);
                    k(h2);
                    if (bVar.f2708a == 4) {
                        i3 += i5;
                    }
                    i5 = 1;
                    m = m2;
                }
            }
            Object obj = bVar.f2710c;
            k(bVar);
            if (i5 > 0) {
                b h3 = h(bVar.f2708a, m, i5, obj);
                e(h3, i3);
                k(h3);
                return;
            }
            return;
        }
        throw new IllegalArgumentException("should not dispatch add or move for pre layout");
    }

    public void e(b bVar, int i) {
        ((RecyclerView.f) this.f2705d).a(bVar);
        int i2 = bVar.f2708a;
        if (i2 != 2) {
            if (i2 == 4) {
                RecyclerView.f fVar = (RecyclerView.f) this.f2705d;
                RecyclerView.this.viewRangeUpdate(i, bVar.f2711d, bVar.f2710c);
                RecyclerView.this.mItemsChanged = true;
                return;
            }
            throw new IllegalArgumentException("only remove and update ops can be dispatched in first pass");
        }
        InterfaceC0053a interfaceC0053a = this.f2705d;
        int i3 = bVar.f2711d;
        RecyclerView.f fVar2 = (RecyclerView.f) interfaceC0053a;
        RecyclerView.this.offsetPositionRecordsForRemove(i, i3, true);
        RecyclerView recyclerView = RecyclerView.this;
        recyclerView.mItemsAddedOrRemoved = true;
        recyclerView.mState.f392c += i3;
    }

    public int f(int i, int i2) {
        int size = this.f2704c.size();
        while (i2 < size) {
            b bVar = this.f2704c.get(i2);
            int i3 = bVar.f2708a;
            if (i3 == 8) {
                int i4 = bVar.f2709b;
                if (i4 == i) {
                    i = bVar.f2711d;
                } else {
                    if (i4 < i) {
                        i--;
                    }
                    if (bVar.f2711d <= i) {
                        i++;
                    }
                }
            } else {
                int i5 = bVar.f2709b;
                if (i5 > i) {
                    continue;
                } else if (i3 == 2) {
                    int i6 = bVar.f2711d;
                    if (i < i5 + i6) {
                        return -1;
                    }
                    i -= i6;
                } else if (i3 == 1) {
                    i += bVar.f2711d;
                }
            }
            i2++;
        }
        return i;
    }

    public boolean g() {
        return this.f2703b.size() > 0;
    }

    public b h(int i, int i2, int i3, Object obj) {
        b b2 = this.f2702a.b();
        if (b2 == null) {
            return new b(i, i2, i3, obj);
        }
        b2.f2708a = i;
        b2.f2709b = i2;
        b2.f2711d = i3;
        b2.f2710c = obj;
        return b2;
    }

    public final void i(b bVar) {
        this.f2704c.add(bVar);
        int i = bVar.f2708a;
        if (i == 1) {
            InterfaceC0053a interfaceC0053a = this.f2705d;
            RecyclerView.f fVar = (RecyclerView.f) interfaceC0053a;
            RecyclerView.this.offsetPositionRecordsForInsert(bVar.f2709b, bVar.f2711d);
            RecyclerView.this.mItemsAddedOrRemoved = true;
        } else if (i == 2) {
            InterfaceC0053a interfaceC0053a2 = this.f2705d;
            RecyclerView.f fVar2 = (RecyclerView.f) interfaceC0053a2;
            RecyclerView.this.offsetPositionRecordsForRemove(bVar.f2709b, bVar.f2711d, false);
            RecyclerView.this.mItemsAddedOrRemoved = true;
        } else if (i == 4) {
            ((RecyclerView.f) this.f2705d).c(bVar.f2709b, bVar.f2711d, bVar.f2710c);
        } else if (i == 8) {
            InterfaceC0053a interfaceC0053a3 = this.f2705d;
            RecyclerView.f fVar3 = (RecyclerView.f) interfaceC0053a3;
            RecyclerView.this.offsetPositionRecordsForMove(bVar.f2709b, bVar.f2711d);
            RecyclerView.this.mItemsAddedOrRemoved = true;
        } else {
            throw new IllegalArgumentException("Unknown update op type for " + bVar);
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:180:0x00a3 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:181:0x0126 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:185:0x0117 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:186:0x00d1 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:198:0x0009 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:29:0x006b  */
    /* JADX WARN: Removed duplicated region for block: B:30:0x0070  */
    /* JADX WARN: Removed duplicated region for block: B:35:0x008e  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x0092  */
    /* JADX WARN: Removed duplicated region for block: B:38:0x009e  */
    /* JADX WARN: Removed duplicated region for block: B:59:0x00d6  */
    /* JADX WARN: Removed duplicated region for block: B:66:0x00f9  */
    /* JADX WARN: Removed duplicated region for block: B:67:0x00fe  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void j() {
        boolean z;
        boolean z2;
        b h2;
        int i;
        int i2;
        boolean z3;
        boolean z4;
        int i3;
        int i4;
        int i5;
        p pVar = this.f2706e;
        ArrayList<b> arrayList = this.f2703b;
        Objects.requireNonNull(pVar);
        while (true) {
            int size = arrayList.size() - 1;
            boolean z5 = false;
            while (true) {
                if (size < 0) {
                    size = -1;
                    break;
                }
                if (arrayList.get(size).f2708a != 8) {
                    z5 = true;
                } else if (z5) {
                    break;
                }
                size--;
            }
            b bVar = null;
            if (size == -1) {
                break;
            }
            int i6 = size + 1;
            b bVar2 = arrayList.get(size);
            b bVar3 = arrayList.get(i6);
            int i7 = bVar3.f2708a;
            if (i7 == 1) {
                int i8 = bVar2.f2711d;
                int i9 = bVar3.f2709b;
                int i10 = i8 < i9 ? -1 : 0;
                int i11 = bVar2.f2709b;
                if (i11 < i9) {
                    i10++;
                }
                if (i9 <= i11) {
                    bVar2.f2709b = i11 + bVar3.f2711d;
                }
                int i12 = bVar3.f2709b;
                if (i12 <= i8) {
                    bVar2.f2711d = i8 + bVar3.f2711d;
                }
                bVar3.f2709b = i12 + i10;
                arrayList.set(size, bVar3);
                arrayList.set(i6, bVar2);
            } else if (i7 == 2) {
                int i13 = bVar2.f2709b;
                int i14 = bVar2.f2711d;
                if (i13 < i14) {
                    if (bVar3.f2709b != i13 || bVar3.f2711d != i14 - i13) {
                        z4 = false;
                        z3 = z4;
                        i3 = bVar3.f2709b;
                        if (i14 >= i3) {
                            bVar3.f2709b = i3 - 1;
                        } else {
                            int i15 = bVar3.f2711d;
                            if (i14 < i3 + i15) {
                                bVar3.f2711d = i15 - 1;
                                bVar2.f2708a = 2;
                                bVar2.f2711d = 1;
                                if (bVar3.f2711d == 0) {
                                    arrayList.remove(i6);
                                    ((a) pVar.f2793a).k(bVar3);
                                }
                            }
                        }
                        i4 = bVar2.f2709b;
                        i5 = bVar3.f2709b;
                        if (i4 > i5) {
                            bVar3.f2709b = i5 + 1;
                        } else {
                            int i16 = i5 + bVar3.f2711d;
                            if (i4 < i16) {
                                bVar = ((a) pVar.f2793a).h(2, i4 + 1, i16 - i4, null);
                                bVar3.f2711d = bVar2.f2709b - bVar3.f2709b;
                            }
                        }
                        if (!z4) {
                            arrayList.set(size, bVar3);
                            arrayList.remove(i6);
                            ((a) pVar.f2793a).k(bVar2);
                        } else {
                            if (z3) {
                                if (bVar != null) {
                                    int i17 = bVar2.f2709b;
                                    if (i17 > bVar.f2709b) {
                                        bVar2.f2709b = i17 - bVar.f2711d;
                                    }
                                    int i18 = bVar2.f2711d;
                                    if (i18 > bVar.f2709b) {
                                        bVar2.f2711d = i18 - bVar.f2711d;
                                    }
                                }
                                int i19 = bVar2.f2709b;
                                if (i19 > bVar3.f2709b) {
                                    bVar2.f2709b = i19 - bVar3.f2711d;
                                }
                                int i20 = bVar2.f2711d;
                                if (i20 > bVar3.f2709b) {
                                    bVar2.f2711d = i20 - bVar3.f2711d;
                                }
                            } else {
                                if (bVar != null) {
                                    int i21 = bVar2.f2709b;
                                    if (i21 >= bVar.f2709b) {
                                        bVar2.f2709b = i21 - bVar.f2711d;
                                    }
                                    int i22 = bVar2.f2711d;
                                    if (i22 >= bVar.f2709b) {
                                        bVar2.f2711d = i22 - bVar.f2711d;
                                    }
                                }
                                int i23 = bVar2.f2709b;
                                if (i23 >= bVar3.f2709b) {
                                    bVar2.f2709b = i23 - bVar3.f2711d;
                                }
                                int i24 = bVar2.f2711d;
                                if (i24 >= bVar3.f2709b) {
                                    bVar2.f2711d = i24 - bVar3.f2711d;
                                }
                            }
                            arrayList.set(size, bVar3);
                            if (bVar2.f2709b != bVar2.f2711d) {
                                arrayList.set(i6, bVar2);
                            } else {
                                arrayList.remove(i6);
                            }
                            if (bVar != null) {
                                arrayList.add(size, bVar);
                            }
                        }
                    } else {
                        z4 = true;
                        z3 = false;
                        i3 = bVar3.f2709b;
                        if (i14 >= i3) {
                        }
                        i4 = bVar2.f2709b;
                        i5 = bVar3.f2709b;
                        if (i4 > i5) {
                        }
                        if (!z4) {
                        }
                    }
                } else if (bVar3.f2709b != i14 + 1 || bVar3.f2711d != i13 - i14) {
                    z3 = true;
                    z4 = false;
                    i3 = bVar3.f2709b;
                    if (i14 >= i3) {
                    }
                    i4 = bVar2.f2709b;
                    i5 = bVar3.f2709b;
                    if (i4 > i5) {
                    }
                    if (!z4) {
                    }
                } else {
                    z4 = true;
                    z3 = z4;
                    i3 = bVar3.f2709b;
                    if (i14 >= i3) {
                    }
                    i4 = bVar2.f2709b;
                    i5 = bVar3.f2709b;
                    if (i4 > i5) {
                    }
                    if (!z4) {
                    }
                }
            } else if (i7 == 4) {
                int i25 = bVar2.f2711d;
                int i26 = bVar3.f2709b;
                if (i25 < i26) {
                    bVar3.f2709b = i26 - 1;
                } else {
                    int i27 = bVar3.f2711d;
                    if (i25 < i26 + i27) {
                        bVar3.f2711d = i27 - 1;
                        h2 = ((a) pVar.f2793a).h(4, bVar2.f2709b, 1, bVar3.f2710c);
                        i = bVar2.f2709b;
                        i2 = bVar3.f2709b;
                        if (i > i2) {
                            bVar3.f2709b = i2 + 1;
                        } else {
                            int i28 = i2 + bVar3.f2711d;
                            if (i < i28) {
                                int i29 = i28 - i;
                                bVar = ((a) pVar.f2793a).h(4, i + 1, i29, bVar3.f2710c);
                                bVar3.f2711d -= i29;
                            }
                        }
                        arrayList.set(i6, bVar2);
                        if (bVar3.f2711d <= 0) {
                            arrayList.set(size, bVar3);
                        } else {
                            arrayList.remove(size);
                            ((a) pVar.f2793a).k(bVar3);
                        }
                        if (h2 != null) {
                            arrayList.add(size, h2);
                        }
                        if (bVar == null) {
                            arrayList.add(size, bVar);
                        }
                    }
                }
                h2 = null;
                i = bVar2.f2709b;
                i2 = bVar3.f2709b;
                if (i > i2) {
                }
                arrayList.set(i6, bVar2);
                if (bVar3.f2711d <= 0) {
                }
                if (h2 != null) {
                }
                if (bVar == null) {
                }
            }
        }
        int size2 = this.f2703b.size();
        for (int i30 = 0; i30 < size2; i30++) {
            b bVar4 = this.f2703b.get(i30);
            int i31 = bVar4.f2708a;
            if (i31 == 1) {
                i(bVar4);
            } else if (i31 == 2) {
                int i32 = bVar4.f2709b;
                int i33 = bVar4.f2711d + i32;
                int i34 = i32;
                boolean z6 = true;
                int i35 = 0;
                while (i34 < i33) {
                    if (((RecyclerView.f) this.f2705d).b(i34) != null || a(i34)) {
                        if (z6) {
                            z = false;
                        } else {
                            d(h(2, i32, i35, null));
                            z = true;
                        }
                        z2 = true;
                    } else {
                        if (z6) {
                            i(h(2, i32, i35, null));
                            z = true;
                        } else {
                            z = false;
                        }
                        z2 = false;
                    }
                    if (z) {
                        i34 -= i35;
                        i33 -= i35;
                        i35 = 1;
                    } else {
                        i35++;
                    }
                    i34++;
                    z6 = z2;
                }
                if (i35 != bVar4.f2711d) {
                    k(bVar4);
                    bVar4 = h(2, i32, i35, null);
                }
                if (!z6) {
                    d(bVar4);
                } else {
                    i(bVar4);
                }
            } else if (i31 == 4) {
                int i36 = bVar4.f2709b;
                int i37 = bVar4.f2711d + i36;
                boolean z7 = true;
                int i38 = i36;
                int i39 = 0;
                while (i36 < i37) {
                    if (((RecyclerView.f) this.f2705d).b(i36) != null || a(i36)) {
                        if (!z7) {
                            d(h(4, i38, i39, bVar4.f2710c));
                            i38 = i36;
                            i39 = 0;
                        }
                        z7 = true;
                    } else {
                        if (z7) {
                            i(h(4, i38, i39, bVar4.f2710c));
                            i38 = i36;
                            i39 = 0;
                        }
                        z7 = false;
                    }
                    i39++;
                    i36++;
                }
                if (i39 != bVar4.f2711d) {
                    Object obj = bVar4.f2710c;
                    k(bVar4);
                    bVar4 = h(4, i38, i39, obj);
                }
                if (!z7) {
                    d(bVar4);
                } else {
                    i(bVar4);
                }
            } else if (i31 == 8) {
                i(bVar4);
            }
        }
        this.f2703b.clear();
    }

    public void k(b bVar) {
        bVar.f2710c = null;
        this.f2702a.a(bVar);
    }

    public void l(List<b> list) {
        int size = list.size();
        for (int i = 0; i < size; i++) {
            k(list.get(i));
        }
        list.clear();
    }

    public final int m(int i, int i2) {
        int i3;
        int i4;
        for (int size = this.f2704c.size() - 1; size >= 0; size--) {
            b bVar = this.f2704c.get(size);
            int i5 = bVar.f2708a;
            if (i5 == 8) {
                int i6 = bVar.f2709b;
                int i7 = bVar.f2711d;
                if (i6 < i7) {
                    i4 = i6;
                    i3 = i7;
                } else {
                    i3 = i6;
                    i4 = i7;
                }
                if (i < i4 || i > i3) {
                    if (i < i6) {
                        if (i2 == 1) {
                            bVar.f2709b = i6 + 1;
                            bVar.f2711d = i7 + 1;
                        } else if (i2 == 2) {
                            bVar.f2709b = i6 - 1;
                            bVar.f2711d = i7 - 1;
                        }
                    }
                } else if (i4 == i6) {
                    if (i2 == 1) {
                        bVar.f2711d = i7 + 1;
                    } else if (i2 == 2) {
                        bVar.f2711d = i7 - 1;
                    }
                    i++;
                } else {
                    if (i2 == 1) {
                        bVar.f2709b = i6 + 1;
                    } else if (i2 == 2) {
                        bVar.f2709b = i6 - 1;
                    }
                    i--;
                }
            } else {
                int i8 = bVar.f2709b;
                if (i8 <= i) {
                    if (i5 == 1) {
                        i -= bVar.f2711d;
                    } else if (i5 == 2) {
                        i += bVar.f2711d;
                    }
                } else if (i2 == 1) {
                    bVar.f2709b = i8 + 1;
                } else if (i2 == 2) {
                    bVar.f2709b = i8 - 1;
                }
            }
        }
        for (int size2 = this.f2704c.size() - 1; size2 >= 0; size2--) {
            b bVar2 = this.f2704c.get(size2);
            if (bVar2.f2708a == 8) {
                int i9 = bVar2.f2711d;
                if (i9 == bVar2.f2709b || i9 < 0) {
                    this.f2704c.remove(size2);
                    k(bVar2);
                }
            } else if (bVar2.f2711d <= 0) {
                this.f2704c.remove(size2);
                k(bVar2);
            }
        }
        return i;
    }
}