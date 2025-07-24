package c.c.a.m.v.c0;

/* compiled from: ByteArrayAdapter.java */
/* loaded from: classes.dex */
public final class f implements a<byte[]> {
    @Override // c.c.a.m.v.c0.a
    public int a() {
        return 1;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.v.c0.a
    public int b(byte[] bArr) {
        return bArr.length;
    }

    @Override // c.c.a.m.v.c0.a
    public String getTag() {
        return "ByteArrayPool";
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.c.a.m.v.c0.a
    public byte[] newArray(int i) {
        return new byte[i];
    }
}