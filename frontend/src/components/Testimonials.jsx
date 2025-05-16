import { cn } from "../lib/utils"
import { Marquee } from "./magicui/marquee"

const reviews = [
  {
    name: "Dr. Jack",
    username: "@jackdoc",
    body: "This AI tool is a game changer for medical imaging analysis. It has significantly improved our diagnostic accuracy.",
    img: "https://avatar.vercel.sh/jack",
  },
  {
    name: "Dr. Jill",
    username: "@jilldoc",
    body: "I'm blown away by how precise and fast this AI is. It has made a huge difference in our workflow.",
    img: "https://avatar.vercel.sh/jill",
  },
  {
    name: "Dr. John",
    username: "@johndoc",
    body: "The AI's ability to analyze medical images is exceptional. It's an invaluable tool for clinicians.",
    img: "https://avatar.vercel.sh/john",
  },
  {
    name: "Dr. Jane",
    username: "@janedoc",
    body: "I highly recommend this tool. It has saved us time and improved the accuracy of our diagnostics.",
    img: "https://avatar.vercel.sh/jane",
  },
  {
    name: "Dr. Jenny",
    username: "@jennydoc",
    body: "The AI's insights have been instrumental in identifying critical patterns in medical images that we missed.",
    img: "https://avatar.vercel.sh/jenny",
  },
  {
    name: "Dr. James",
    username: "@jamesdoc",
    body: "I've never seen such a powerful AI for medical image analysis. It's a must-have in any healthcare facility.",
    img: "https://avatar.vercel.sh/james",
  },
];

const firstRow = reviews.slice(0, reviews.length / 2);

const ReviewCard = ({
  img,
  name,
  username,
  body,
}) => {
  return (
    <figure
      className={cn(
        "relative h-full w-64 cursor-pointer overflow-hidden rounded-xl border p-4",
        // light styles
        "border-gray-950/[.1] bg-gray-950/[.01] hover:bg-gray-950/[.05]",
        // dark styles
        "dark:border-gray-50/[.1] dark:bg-gray-50/[.10] dark:hover:bg-gray-50/[.15]",
      )}
    >
      <div className="flex flex-row items-center gap-2">
        <img className="rounded-full" width="32" height="32" alt="" src={img} />
        <div className="flex flex-col">
          <figcaption className="text-sm font-medium dark:text-white">
            {name}
          </figcaption>
          <p className="text-xs font-medium dark:text-white/40">{username}</p>
        </div>
      </div>
      <blockquote className="mt-2 text-sm">{body}</blockquote>
    </figure>
  );
};

export default function Testimonials() {
  return (
    <div className="relative flex w-full flex-col items-center justify-center overflow-hidden bg-white dark:bg-zinc-950">

      <Marquee pauseOnHover className="[--duration:20s]">
        {firstRow.map((review) => (
          <ReviewCard key={review.username} {...review} />
        ))}
      </Marquee>
      
      <div className="pointer-events-none absolute inset-y-0 left-0 w-1/4 bg-gradient-to-r from-background"></div>
      <div className="pointer-events-none absolute inset-y-0 right-0 w-1/4 bg-gradient-to-l from-background"></div>
    </div>
  );
}
