package android.support.v4.media;

import android.os.Parcel;
import android.os.Parcelable;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* loaded from: classes.dex */
public final class RatingCompat implements Parcelable {
    public static final Parcelable.Creator<RatingCompat> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final int f15b;

    /* renamed from: c  reason: collision with root package name */
    public final float f16c;

    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<RatingCompat> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public RatingCompat createFromParcel(Parcel parcel) {
            return new RatingCompat(parcel.readInt(), parcel.readFloat());
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public RatingCompat[] newArray(int i) {
            return new RatingCompat[i];
        }
    }

    public RatingCompat(int i, float f2) {
        this.f15b = i;
        this.f16c = f2;
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return this.f15b;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Rating:style=");
        x.append(this.f15b);
        x.append(" rating=");
        float f2 = this.f16c;
        x.append(f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? "unrated" : String.valueOf(f2));
        return x.toString();
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeInt(this.f15b);
        parcel.writeFloat(this.f16c);
    }
}