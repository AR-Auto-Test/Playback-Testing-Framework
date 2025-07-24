package b.b0;

import android.os.Parcel;
import android.os.Parcelable;
import android.text.TextUtils;
import android.util.SparseIntArray;
import java.lang.reflect.Method;

/* compiled from: VersionedParcelParcel.java */
/* loaded from: classes.dex */
public class b extends a {

    /* renamed from: d  reason: collision with root package name */
    public final SparseIntArray f979d;

    /* renamed from: e  reason: collision with root package name */
    public final Parcel f980e;

    /* renamed from: f  reason: collision with root package name */
    public final int f981f;

    /* renamed from: g  reason: collision with root package name */
    public final int f982g;

    /* renamed from: h  reason: collision with root package name */
    public final String f983h;
    public int i;
    public int j;
    public int k;

    public b(Parcel parcel) {
        this(parcel, parcel.dataPosition(), parcel.dataSize(), "", new b.f.a(), new b.f.a(), new b.f.a());
    }

    @Override // b.b0.a
    public void a() {
        int i = this.i;
        if (i >= 0) {
            int i2 = this.f979d.get(i);
            int dataPosition = this.f980e.dataPosition();
            this.f980e.setDataPosition(i2);
            this.f980e.writeInt(dataPosition - i2);
            this.f980e.setDataPosition(dataPosition);
        }
    }

    @Override // b.b0.a
    public a b() {
        Parcel parcel = this.f980e;
        int dataPosition = parcel.dataPosition();
        int i = this.j;
        if (i == this.f981f) {
            i = this.f982g;
        }
        return new b(parcel, dataPosition, i, c.b.a.a.a.v(new StringBuilder(), this.f983h, "  "), this.f976a, this.f977b, this.f978c);
    }

    @Override // b.b0.a
    public boolean f() {
        return this.f980e.readInt() != 0;
    }

    @Override // b.b0.a
    public byte[] g() {
        int readInt = this.f980e.readInt();
        if (readInt < 0) {
            return null;
        }
        byte[] bArr = new byte[readInt];
        this.f980e.readByteArray(bArr);
        return bArr;
    }

    @Override // b.b0.a
    public CharSequence h() {
        return (CharSequence) TextUtils.CHAR_SEQUENCE_CREATOR.createFromParcel(this.f980e);
    }

    @Override // b.b0.a
    public boolean i(int i) {
        while (this.j < this.f982g) {
            int i2 = this.k;
            if (i2 == i) {
                return true;
            }
            if (String.valueOf(i2).compareTo(String.valueOf(i)) > 0) {
                return false;
            }
            this.f980e.setDataPosition(this.j);
            int readInt = this.f980e.readInt();
            this.k = this.f980e.readInt();
            this.j += readInt;
        }
        return this.k == i;
    }

    @Override // b.b0.a
    public int j() {
        return this.f980e.readInt();
    }

    @Override // b.b0.a
    public <T extends Parcelable> T l() {
        return (T) this.f980e.readParcelable(b.class.getClassLoader());
    }

    @Override // b.b0.a
    public String n() {
        return this.f980e.readString();
    }

    @Override // b.b0.a
    public void p(int i) {
        a();
        this.i = i;
        this.f979d.put(i, this.f980e.dataPosition());
        this.f980e.writeInt(0);
        this.f980e.writeInt(i);
    }

    @Override // b.b0.a
    public void q(boolean z) {
        this.f980e.writeInt(z ? 1 : 0);
    }

    @Override // b.b0.a
    public void r(byte[] bArr) {
        if (bArr != null) {
            this.f980e.writeInt(bArr.length);
            this.f980e.writeByteArray(bArr);
            return;
        }
        this.f980e.writeInt(-1);
    }

    @Override // b.b0.a
    public void s(CharSequence charSequence) {
        TextUtils.writeToParcel(charSequence, this.f980e, 0);
    }

    @Override // b.b0.a
    public void t(int i) {
        this.f980e.writeInt(i);
    }

    @Override // b.b0.a
    public void u(Parcelable parcelable) {
        this.f980e.writeParcelable(parcelable, 0);
    }

    @Override // b.b0.a
    public void v(String str) {
        this.f980e.writeString(str);
    }

    public b(Parcel parcel, int i, int i2, String str, b.f.a<String, Method> aVar, b.f.a<String, Method> aVar2, b.f.a<String, Class> aVar3) {
        super(aVar, aVar2, aVar3);
        this.f979d = new SparseIntArray();
        this.i = -1;
        this.j = 0;
        this.k = -1;
        this.f980e = parcel;
        this.f981f = i;
        this.f982g = i2;
        this.j = i;
        this.f983h = str;
    }
}