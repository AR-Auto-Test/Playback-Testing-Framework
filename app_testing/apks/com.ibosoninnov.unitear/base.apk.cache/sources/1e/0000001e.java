package android.support.v4.media.session;

import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.text.TextUtils;
import java.util.List;

/* loaded from: classes.dex */
public final class PlaybackStateCompat implements Parcelable {
    public static final Parcelable.Creator<PlaybackStateCompat> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final int f27b;

    /* renamed from: c  reason: collision with root package name */
    public final long f28c;

    /* renamed from: d  reason: collision with root package name */
    public final long f29d;

    /* renamed from: e  reason: collision with root package name */
    public final float f30e;

    /* renamed from: f  reason: collision with root package name */
    public final long f31f;

    /* renamed from: g  reason: collision with root package name */
    public final int f32g;

    /* renamed from: h  reason: collision with root package name */
    public final CharSequence f33h;
    public final long i;
    public List<CustomAction> j;
    public final long k;
    public final Bundle l;

    /* loaded from: classes.dex */
    public static final class CustomAction implements Parcelable {
        public static final Parcelable.Creator<CustomAction> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public final String f34b;

        /* renamed from: c  reason: collision with root package name */
        public final CharSequence f35c;

        /* renamed from: d  reason: collision with root package name */
        public final int f36d;

        /* renamed from: e  reason: collision with root package name */
        public final Bundle f37e;

        /* loaded from: classes.dex */
        public static class a implements Parcelable.Creator<CustomAction> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public CustomAction createFromParcel(Parcel parcel) {
                return new CustomAction(parcel);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public CustomAction[] newArray(int i) {
                return new CustomAction[i];
            }
        }

        public CustomAction(Parcel parcel) {
            this.f34b = parcel.readString();
            this.f35c = (CharSequence) TextUtils.CHAR_SEQUENCE_CREATOR.createFromParcel(parcel);
            this.f36d = parcel.readInt();
            this.f37e = parcel.readBundle(MediaSessionCompat.class.getClassLoader());
        }

        @Override // android.os.Parcelable
        public int describeContents() {
            return 0;
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("Action:mName='");
            x.append((Object) this.f35c);
            x.append(", mIcon=");
            x.append(this.f36d);
            x.append(", mExtras=");
            x.append(this.f37e);
            return x.toString();
        }

        @Override // android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            parcel.writeString(this.f34b);
            TextUtils.writeToParcel(this.f35c, parcel, i);
            parcel.writeInt(this.f36d);
            parcel.writeBundle(this.f37e);
        }
    }

    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<PlaybackStateCompat> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public PlaybackStateCompat createFromParcel(Parcel parcel) {
            return new PlaybackStateCompat(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public PlaybackStateCompat[] newArray(int i) {
            return new PlaybackStateCompat[i];
        }
    }

    public PlaybackStateCompat(Parcel parcel) {
        this.f27b = parcel.readInt();
        this.f28c = parcel.readLong();
        this.f30e = parcel.readFloat();
        this.i = parcel.readLong();
        this.f29d = parcel.readLong();
        this.f31f = parcel.readLong();
        this.f33h = (CharSequence) TextUtils.CHAR_SEQUENCE_CREATOR.createFromParcel(parcel);
        this.j = parcel.createTypedArrayList(CustomAction.CREATOR);
        this.k = parcel.readLong();
        this.l = parcel.readBundle(MediaSessionCompat.class.getClassLoader());
        this.f32g = parcel.readInt();
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    public String toString() {
        return "PlaybackState {state=" + this.f27b + ", position=" + this.f28c + ", buffered position=" + this.f29d + ", speed=" + this.f30e + ", updated=" + this.i + ", actions=" + this.f31f + ", error code=" + this.f32g + ", error message=" + this.f33h + ", custom actions=" + this.j + ", active item id=" + this.k + "}";
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeInt(this.f27b);
        parcel.writeLong(this.f28c);
        parcel.writeFloat(this.f30e);
        parcel.writeLong(this.i);
        parcel.writeLong(this.f29d);
        parcel.writeLong(this.f31f);
        TextUtils.writeToParcel(this.f33h, parcel, i);
        parcel.writeTypedList(this.j);
        parcel.writeLong(this.k);
        parcel.writeBundle(this.l);
        parcel.writeInt(this.f32g);
    }
}