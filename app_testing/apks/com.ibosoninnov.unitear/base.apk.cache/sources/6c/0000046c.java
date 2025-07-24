package b.j.h;

/* compiled from: TextDirectionHeuristicsCompat.java */
/* loaded from: classes.dex */
public final class d {

    /* renamed from: a  reason: collision with root package name */
    public static final b.j.h.c f2182a = new C0035d(null, false);

    /* renamed from: b  reason: collision with root package name */
    public static final b.j.h.c f2183b = new C0035d(null, true);

    /* renamed from: c  reason: collision with root package name */
    public static final b.j.h.c f2184c;

    /* renamed from: d  reason: collision with root package name */
    public static final b.j.h.c f2185d;

    /* compiled from: TextDirectionHeuristicsCompat.java */
    /* loaded from: classes.dex */
    public static class a implements b {

        /* renamed from: a  reason: collision with root package name */
        public static final a f2186a = new a();

        @Override // b.j.h.d.b
        public int a(CharSequence charSequence, int i, int i2) {
            int i3 = i2 + i;
            int i4 = 2;
            while (i < i3 && i4 == 2) {
                byte directionality = Character.getDirectionality(charSequence.charAt(i));
                b.j.h.c cVar = d.f2182a;
                if (directionality != 0) {
                    if (directionality != 1 && directionality != 2) {
                        switch (directionality) {
                            case 14:
                            case 15:
                                break;
                            case 16:
                            case 17:
                                break;
                            default:
                                i4 = 2;
                                break;
                        }
                        i++;
                    }
                    i4 = 0;
                    i++;
                }
                i4 = 1;
                i++;
            }
            return i4;
        }
    }

    /* compiled from: TextDirectionHeuristicsCompat.java */
    /* loaded from: classes.dex */
    public interface b {
        int a(CharSequence charSequence, int i, int i2);
    }

    /* compiled from: TextDirectionHeuristicsCompat.java */
    /* loaded from: classes.dex */
    public static abstract class c implements b.j.h.c {

        /* renamed from: a  reason: collision with root package name */
        public final b f2187a;

        public c(b bVar) {
            this.f2187a = bVar;
        }

        public abstract boolean a();

        public boolean b(CharSequence charSequence, int i, int i2) {
            if (i >= 0 && i2 >= 0 && charSequence.length() - i2 >= i) {
                b bVar = this.f2187a;
                if (bVar == null) {
                    return a();
                }
                int a2 = bVar.a(charSequence, i, i2);
                if (a2 != 0) {
                    if (a2 != 1) {
                        return a();
                    }
                    return false;
                }
                return true;
            }
            throw new IllegalArgumentException();
        }
    }

    /* compiled from: TextDirectionHeuristicsCompat.java */
    /* renamed from: b.j.h.d$d  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0035d extends c {

        /* renamed from: b  reason: collision with root package name */
        public final boolean f2188b;

        public C0035d(b bVar, boolean z) {
            super(bVar);
            this.f2188b = z;
        }

        @Override // b.j.h.d.c
        public boolean a() {
            return this.f2188b;
        }
    }

    static {
        a aVar = a.f2186a;
        f2184c = new C0035d(aVar, false);
        f2185d = new C0035d(aVar, true);
    }
}