package c.c.a.m.v.c0;

/* compiled from: IntegerArrayAdapter.java */
/* loaded from: classes.dex */
public final class h implements a<int[]> {
    @Override // c.c.a.m.v.c0.a
    public int a() {
        return 4;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.v.c0.a
    public int b(int[] iArr) {
        return iArr.length;
    }

    @Override // c.c.a.m.v.c0.a
    public String getTag() {
        return "IntegerArrayPool";
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.c.a.m.v.c0.a
    public int[] newArray(int i) {
        return new int[i];
    }
}