package b.q.b;

import android.annotation.SuppressLint;
import android.os.Parcel;
import android.os.Parcelable;
import android.text.TextUtils;
import androidx.fragment.app.Fragment;
import b.q.b.y;
import java.util.ArrayList;

/* compiled from: BackStackState.java */
@SuppressLint({"BanParcelableUsage"})
/* loaded from: classes.dex */
public final class b implements Parcelable {
    public static final Parcelable.Creator<b> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final int[] f2398b;

    /* renamed from: c  reason: collision with root package name */
    public final ArrayList<String> f2399c;

    /* renamed from: d  reason: collision with root package name */
    public final int[] f2400d;

    /* renamed from: e  reason: collision with root package name */
    public final int[] f2401e;

    /* renamed from: f  reason: collision with root package name */
    public final int f2402f;

    /* renamed from: g  reason: collision with root package name */
    public final String f2403g;

    /* renamed from: h  reason: collision with root package name */
    public final int f2404h;
    public final int i;
    public final CharSequence j;
    public final int k;
    public final CharSequence l;
    public final ArrayList<String> m;
    public final ArrayList<String> n;
    public final boolean o;

    /* compiled from: BackStackState.java */
    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<b> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public b createFromParcel(Parcel parcel) {
            return new b(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public b[] newArray(int i) {
            return new b[i];
        }
    }

    public b(b.q.b.a aVar) {
        int size = aVar.f2541a.size();
        this.f2398b = new int[size * 5];
        if (aVar.f2547g) {
            this.f2399c = new ArrayList<>(size);
            this.f2400d = new int[size];
            this.f2401e = new int[size];
            int i = 0;
            int i2 = 0;
            while (i < size) {
                y.a aVar2 = aVar.f2541a.get(i);
                int i3 = i2 + 1;
                this.f2398b[i2] = aVar2.f2549a;
                ArrayList<String> arrayList = this.f2399c;
                Fragment fragment = aVar2.f2550b;
                arrayList.add(fragment != null ? fragment.mWho : null);
                int[] iArr = this.f2398b;
                int i4 = i3 + 1;
                iArr[i3] = aVar2.f2551c;
                int i5 = i4 + 1;
                iArr[i4] = aVar2.f2552d;
                int i6 = i5 + 1;
                iArr[i5] = aVar2.f2553e;
                iArr[i6] = aVar2.f2554f;
                this.f2400d[i] = aVar2.f2555g.ordinal();
                this.f2401e[i] = aVar2.f2556h.ordinal();
                i++;
                i2 = i6 + 1;
            }
            this.f2402f = aVar.f2546f;
            this.f2403g = aVar.i;
            this.f2404h = aVar.s;
            this.i = aVar.j;
            this.j = aVar.k;
            this.k = aVar.l;
            this.l = aVar.m;
            this.m = aVar.n;
            this.n = aVar.o;
            this.o = aVar.p;
            return;
        }
        throw new IllegalStateException("Not on back stack");
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeIntArray(this.f2398b);
        parcel.writeStringList(this.f2399c);
        parcel.writeIntArray(this.f2400d);
        parcel.writeIntArray(this.f2401e);
        parcel.writeInt(this.f2402f);
        parcel.writeString(this.f2403g);
        parcel.writeInt(this.f2404h);
        parcel.writeInt(this.i);
        TextUtils.writeToParcel(this.j, parcel, 0);
        parcel.writeInt(this.k);
        TextUtils.writeToParcel(this.l, parcel, 0);
        parcel.writeStringList(this.m);
        parcel.writeStringList(this.n);
        parcel.writeInt(this.o ? 1 : 0);
    }

    public b(Parcel parcel) {
        this.f2398b = parcel.createIntArray();
        this.f2399c = parcel.createStringArrayList();
        this.f2400d = parcel.createIntArray();
        this.f2401e = parcel.createIntArray();
        this.f2402f = parcel.readInt();
        this.f2403g = parcel.readString();
        this.f2404h = parcel.readInt();
        this.i = parcel.readInt();
        this.j = (CharSequence) TextUtils.CHAR_SEQUENCE_CREATOR.createFromParcel(parcel);
        this.k = parcel.readInt();
        this.l = (CharSequence) TextUtils.CHAR_SEQUENCE_CREATOR.createFromParcel(parcel);
        this.m = parcel.createStringArrayList();
        this.n = parcel.createStringArrayList();
        this.o = parcel.readInt() != 0;
    }
}