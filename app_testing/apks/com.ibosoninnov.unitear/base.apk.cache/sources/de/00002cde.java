package e;

import java.util.Arrays;

/* compiled from: KotlinVersion.kt */
/* loaded from: classes2.dex */
public final class a implements Comparable<a> {

    /* renamed from: b  reason: collision with root package name */
    public static final a f5710b = new a(1, 4, 10);

    /* renamed from: c  reason: collision with root package name */
    public final int f5711c;

    /* renamed from: d  reason: collision with root package name */
    public final int f5712d;

    /* renamed from: e  reason: collision with root package name */
    public final int f5713e;

    /* renamed from: f  reason: collision with root package name */
    public final int f5714f;

    public a(int i, int i2, int i3) {
        this.f5712d = i;
        this.f5713e = i2;
        this.f5714f = i3;
        if (i >= 0 && 255 >= i && i2 >= 0 && 255 >= i2 && i3 >= 0 && 255 >= i3) {
            this.f5711c = (i << 16) + (i2 << 8) + i3;
            return;
        }
        throw new IllegalArgumentException(("Version components are out of range: " + i + '.' + i2 + '.' + i3).toString());
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // java.lang.Comparable
    public int compareTo(a aVar) {
        a aVar2 = aVar;
        if (aVar2 == null) {
            StackTraceElement stackTraceElement = Thread.currentThread().getStackTrace()[4];
            String className = stackTraceElement.getClassName();
            String methodName = stackTraceElement.getMethodName();
            NullPointerException nullPointerException = new NullPointerException("Parameter specified as non-null is null: method " + className + "." + methodName + ", parameter other");
            String name = e.b.a.a.class.getName();
            StackTraceElement[] stackTrace = nullPointerException.getStackTrace();
            int length = stackTrace.length;
            int i = -1;
            for (int i2 = 0; i2 < length; i2++) {
                if (name.equals(stackTrace[i2].getClassName())) {
                    i = i2;
                }
            }
            nullPointerException.setStackTrace((StackTraceElement[]) Arrays.copyOfRange(stackTrace, i + 1, length));
            throw nullPointerException;
        }
        return this.f5711c - aVar2.f5711c;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof a)) {
            obj = null;
        }
        a aVar = (a) obj;
        return aVar != null && this.f5711c == aVar.f5711c;
    }

    public int hashCode() {
        return this.f5711c;
    }

    /* JADX DEBUG: TODO: convert one arg to string using `String.valueOf()`, args: [(wrap: int : 0x0005: IGET  (r1v0 int A[REMOVE]) = (r3v0 'this' e.a A[IMMUTABLE_TYPE, THIS]) e.a.d int), ('.' char), (wrap: int : 0x000f: IGET  (r2v0 int A[REMOVE]) = (r3v0 'this' e.a A[IMMUTABLE_TYPE, THIS]) e.a.e int), ('.' char), (wrap: int : 0x0017: IGET  (r1v2 int A[REMOVE]) = (r3v0 'this' e.a A[IMMUTABLE_TYPE, THIS]) e.a.f int)] */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.f5712d);
        sb.append('.');
        sb.append(this.f5713e);
        sb.append('.');
        sb.append(this.f5714f);
        return sb.toString();
    }
}