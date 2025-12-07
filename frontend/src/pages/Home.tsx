import { Link } from 'react-router-dom';
import { BarChart3, Calculator, MessageSquare, Search, ArrowRight, Laptop, TrendingUp, Users } from 'lucide-react';

const features = [
  {
    category: 'Descriptive',
    color: 'green',
    items: [
      {
        title: 'Market Overview',
        description: 'Statistical summary and trends of the laptop market',
        icon: BarChart3,
        link: '/market',
        features: ['Price distributions', 'Brand analysis', 'Feature correlations'],
      },
      {
        title: 'Market Segmentation',
        description: 'Clustering analysis of different laptop types',
        icon: Users,
        link: '/market',
        features: ['5 market segments', 'Segment profiles', 'Visual cluster mapping'],
      },
    ],
  },
  {
    category: 'Predictive',
    color: 'blue',
    items: [
      {
        title: 'Price Predictor',
        description: 'Get price estimates with feature importance breakdown',
        icon: Calculator,
        link: '/predict',
        features: ['Simple & Advanced modes', '30+ features', 'Price breakdown'],
      },
      {
        title: 'Chat Advisor',
        description: 'AI-powered conversational assistant',
        icon: MessageSquare,
        link: '/chat',
        features: ['Natural language', 'Personalized recommendations', 'Budget-aware'],
      },
    ],
  },
  {
    category: 'Prescriptive',
    color: 'amber',
    items: [
      {
        title: 'Find Similar',
        description: 'Discover laptops matching your requirements',
        icon: Search,
        link: '/similar',
        features: ['K-best matches', 'Distance comparison', 'Price vs specs'],
      },
    ],
  },
];

const stats = [
  { label: 'Laptops Analyzed', value: '5,915' },
  { label: 'Price Range', value: '€0 - €3,605' },
  { label: 'Model Accuracy', value: '±20%' },
  { label: 'Market Segments', value: '5' },
];

export default function Home() {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <div className="flex justify-center">
          <div className="bg-primary-100 p-4 rounded-2xl">
            <Laptop className="h-16 w-16 text-primary-600" />
          </div>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900">
          Computer Price Predictor
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Explore the laptop market, predict prices, and find your perfect match
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="card text-center">
            <div className="text-2xl font-bold text-primary-600">{stat.value}</div>
            <div className="text-sm text-gray-500">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Feature Sections */}
      {features.map((section) => (
        <div key={section.category} className="space-y-4">
          <div className="flex items-center gap-2">
            <TrendingUp className={`h-5 w-5 text-${section.color}-500`} />
            <h2 className="text-xl font-semibold text-gray-900">
              {section.category} Analytics
            </h2>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {section.items.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.title}
                  to={item.link}
                  className="card hover:shadow-md transition-shadow group"
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl bg-${section.color}-100`}>
                      <Icon className={`h-6 w-6 text-${section.color}-600`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-gray-900">{item.title}</h3>
                        <ArrowRight className="h-5 w-5 text-gray-400 group-hover:text-primary-500 transition-colors" />
                      </div>
                      <p className="text-sm text-gray-600 mt-1">{item.description}</p>
                      <ul className="mt-3 space-y-1">
                        {item.features.map((feature) => (
                          <li key={feature} className="text-sm text-gray-500 flex items-center gap-2">
                            <span className={`w-1.5 h-1.5 rounded-full bg-${section.color}-400`} />
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      ))}

      {/* Quick Start */}
      <div className="card bg-gradient-to-r from-primary-500 to-primary-600 text-white">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-semibold">Ready to get started?</h3>
            <p className="text-primary-100 mt-1">
              Try the Chat Advisor for personalized recommendations
            </p>
          </div>
          <Link
            to="/chat"
            className="flex items-center gap-2 bg-white text-primary-600 px-6 py-3 rounded-lg font-medium hover:bg-primary-50 transition-colors"
          >
            <MessageSquare className="h-5 w-5" />
            Start Chatting
          </Link>
        </div>
      </div>

      {/* Example Use Cases */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Example Use Cases</h2>
        <div className="grid md:grid-cols-4 gap-4">
          {[
            { emoji: '🎮', label: 'Gaming', specs: 'RTX 4060, i7, 16GB', price: '€1,200-1,500' },
            { emoji: '💼', label: 'Work', specs: 'i5, Integrated, 16GB', price: '€700-900' },
            { emoji: '🎨', label: 'Creative', specs: 'M3 Pro, 32GB', price: '€2,000-2,500' },
            { emoji: '📚', label: 'Student', specs: 'i5, 8GB, 256GB', price: '€500-700' },
          ].map((example) => (
            <div key={example.label} className="card">
              <div className="text-2xl mb-2">{example.emoji}</div>
              <div className="font-medium text-gray-900">{example.label}</div>
              <div className="text-sm text-gray-500">{example.specs}</div>
              <div className="text-sm font-medium text-primary-600 mt-1">{example.price}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
