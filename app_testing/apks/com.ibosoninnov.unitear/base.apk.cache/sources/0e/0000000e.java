package android.support.v4.media;

import android.graphics.Bitmap;
import android.media.MediaDescription;
import android.net.Uri;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.support.v4.media.session.MediaSessionCompat;

/* loaded from: classes.dex */
public final class MediaDescriptionCompat implements Parcelable {
    public static final Parcelable.Creator<MediaDescriptionCompat> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final String f6b;

    /* renamed from: c  reason: collision with root package name */
    public final CharSequence f7c;

    /* renamed from: d  reason: collision with root package name */
    public final CharSequence f8d;

    /* renamed from: e  reason: collision with root package name */
    public final CharSequence f9e;

    /* renamed from: f  reason: collision with root package name */
    public final Bitmap f10f;

    /* renamed from: g  reason: collision with root package name */
    public final Uri f11g;

    /* renamed from: h  reason: collision with root package name */
    public final Bundle f12h;
    public final Uri i;
    public Object j;

    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<MediaDescriptionCompat> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public MediaDescriptionCompat createFromParcel(Parcel parcel) {
            return MediaDescriptionCompat.a(MediaDescription.CREATOR.createFromParcel(parcel));
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public MediaDescriptionCompat[] newArray(int i) {
            return new MediaDescriptionCompat[i];
        }
    }

    public MediaDescriptionCompat(String str, CharSequence charSequence, CharSequence charSequence2, CharSequence charSequence3, Bitmap bitmap, Uri uri, Bundle bundle, Uri uri2) {
        this.f6b = str;
        this.f7c = charSequence;
        this.f8d = charSequence2;
        this.f9e = charSequence3;
        this.f10f = bitmap;
        this.f11g = uri;
        this.f12h = bundle;
        this.i = uri2;
    }

    /* JADX WARN: Removed duplicated region for block: B:18:0x004e  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static MediaDescriptionCompat a(Object obj) {
        Uri uri;
        Bundle bundle;
        if (obj != null) {
            MediaDescription mediaDescription = (MediaDescription) obj;
            String mediaId = mediaDescription.getMediaId();
            CharSequence title = mediaDescription.getTitle();
            CharSequence subtitle = mediaDescription.getSubtitle();
            CharSequence description = mediaDescription.getDescription();
            Bitmap iconBitmap = mediaDescription.getIconBitmap();
            Uri iconUri = mediaDescription.getIconUri();
            Bundle extras = mediaDescription.getExtras();
            if (extras != null) {
                MediaSessionCompat.a(extras);
                uri = (Uri) extras.getParcelable("android.support.v4.media.description.MEDIA_URI");
            } else {
                uri = null;
            }
            if (uri != null) {
                if (!extras.containsKey("android.support.v4.media.description.NULL_BUNDLE_FLAG") || extras.size() != 2) {
                    extras.remove("android.support.v4.media.description.MEDIA_URI");
                    extras.remove("android.support.v4.media.description.NULL_BUNDLE_FLAG");
                } else {
                    bundle = null;
                    if (uri == null) {
                        uri = mediaDescription.getMediaUri();
                    }
                    MediaDescriptionCompat mediaDescriptionCompat = new MediaDescriptionCompat(mediaId, title, subtitle, description, iconBitmap, iconUri, bundle, uri);
                    mediaDescriptionCompat.j = obj;
                    return mediaDescriptionCompat;
                }
            }
            bundle = extras;
            if (uri == null) {
            }
            MediaDescriptionCompat mediaDescriptionCompat2 = new MediaDescriptionCompat(mediaId, title, subtitle, description, iconBitmap, iconUri, bundle, uri);
            mediaDescriptionCompat2.j = obj;
            return mediaDescriptionCompat2;
        }
        return null;
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    public String toString() {
        return ((Object) this.f7c) + ", " + ((Object) this.f8d) + ", " + ((Object) this.f9e);
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        Object obj = this.j;
        if (obj == null) {
            MediaDescription.Builder builder = new MediaDescription.Builder();
            builder.setMediaId(this.f6b);
            builder.setTitle(this.f7c);
            builder.setSubtitle(this.f8d);
            builder.setDescription(this.f9e);
            builder.setIconBitmap(this.f10f);
            builder.setIconUri(this.f11g);
            builder.setExtras(this.f12h);
            builder.setMediaUri(this.i);
            obj = builder.build();
            this.j = obj;
        }
        ((MediaDescription) obj).writeToParcel(parcel, i);
    }
}